//! Specialize opaque image/sampler function arguments to their bound globals.
//! This also preserves comparison-sampler typing through the SPIR-V round trip:
//! SPIR-V has no distinct OpTypeSampler for comparison function parameters.
use ash::vk;
use naga::{Expression, Function, GlobalVariable, Handle, Statement, TypeInner};
use std::collections::HashMap;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
type Bindings = Vec<Option<Handle<GlobalVariable>>>;
struct Specializer<'a> {
    source: &'a naga::Arena<Function>,
    types: &'a naga::UniqueArena<naga::Type>,
    output: naga::Arena<Function>,
    cache: HashMap<(Handle<Function>, Bindings), Handle<Function>>,
    active: Vec<Handle<Function>>,
}
fn global(
    expressions: &naga::Arena<Expression>,
    expression: Handle<Expression>,
) -> Result<Handle<GlobalVariable>, vk::Result> {
    match expressions[expression] {
        Expression::GlobalVariable(global) => Ok(global),
        Expression::Load { pointer } => global(expressions, pointer),
        _ => Err(UNSUPPORTED),
    }
}
impl Specializer<'_> {
    fn function(
        &mut self,
        handle: Handle<Function>,
        bindings: Bindings,
    ) -> Result<Handle<Function>, vk::Result> {
        let key = (handle, bindings);
        if let Some(&handle) = self.cache.get(&key) {
            return Ok(handle);
        }
        if self.active.contains(&handle) {
            return Err(UNSUPPORTED);
        }
        self.active.push(handle);
        let mut function = self.source[handle].clone();
        for (_, expr) in function.expressions.iter_mut() {
            if let Expression::FunctionArgument(index) = expr {
                let i = *index as usize;
                if let Some(global) = key.1[i] {
                    *expr = Expression::GlobalVariable(global);
                } else {
                    *index -= key.1[..i].iter().filter(|g| g.is_some()).count() as u32;
                }
            }
        }
        function.arguments = function
            .arguments
            .into_iter()
            .enumerate()
            .filter_map(|(i, arg)| key.1[i].is_none().then_some(arg))
            .collect();
        self.block(&mut function.expressions, &mut function.body)?;
        self.active.pop();
        let handle = self.output.append(function, naga::Span::UNDEFINED);
        self.cache.insert(key, handle);
        Ok(handle)
    }
    fn block(
        &mut self,
        expressions: &mut naga::Arena<Expression>,
        block: &mut naga::Block,
    ) -> Result<(), vk::Result> {
        for statement in block.iter_mut() {
            match statement {
                Statement::Block(block) => self.block(expressions, block)?,
                Statement::If { accept, reject, .. } => {
                    self.block(expressions, accept)?;
                    self.block(expressions, reject)?;
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.block(expressions, &mut case.body)?;
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.block(expressions, body)?;
                    self.block(expressions, continuing)?;
                }
                Statement::Call {
                    function,
                    arguments,
                    result,
                } => {
                    let callee = &self.source[*function];
                    if callee.arguments.len() != arguments.len() {
                        return Err(UNSUPPORTED);
                    }
                    let bindings = callee
                        .arguments
                        .iter()
                        .zip(arguments.iter())
                        .map(|(formal, actual)| {
                            if matches!(
                                self.types[formal.ty].inner,
                                TypeInner::Image { .. } | TypeInner::Sampler { .. }
                            ) {
                                global(expressions, *actual).map(Some)
                            } else {
                                Ok(None)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let retained = arguments
                        .iter()
                        .zip(&bindings)
                        .filter_map(|(arg, bound)| bound.is_none().then_some(*arg))
                        .collect();
                    *function = self.function(*function, bindings)?;
                    *arguments = retained;
                    if let Some(result) = result {
                        expressions[*result] = Expression::CallResult(*function);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
pub(crate) fn specialize(module: &mut naga::Module) -> Result<(), vk::Result> {
    let mut specializer = Specializer {
        source: &module.functions,
        types: &module.types,
        output: Default::default(),
        cache: Default::default(),
        active: Vec::new(),
    };
    for entry in &mut module.entry_points {
        specializer.block(&mut entry.function.expressions, &mut entry.function.body)?;
    }
    module.functions = specializer.output;
    Ok(())
}

// Vulkan permits fewer fragment components than the attachment stores. WGPU
// requires enough components for its target format; fill unspecified channels
// with zero (alpha one) while retaining every value produced by the shader.
pub(crate) fn widen_fragment_outputs(module: &mut naga::Module) -> Result<(), vk::Result> {
    let vec4 = module.types.insert(
        naga::Type {
            name: None,
            inner: TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: naga::Scalar::F32,
            },
        },
        naga::Span::UNDEFINED,
    );
    let width = |ty: &TypeInner| match ty {
        TypeInner::Scalar(naga::Scalar {
            kind: naga::ScalarKind::Float,
            width: 4,
        }) => Some(1),
        TypeInner::Vector {
            size,
            scalar:
                naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4,
                },
        } => Some(*size as u32),
        _ => None,
    };
    for entry in &mut module.entry_points {
        if entry.stage != naga::ShaderStage::Fragment {
            continue;
        }
        let Some(result) = entry.function.result.as_mut() else {
            continue;
        };
        let (new_type, members) = match &module.types[result.ty].inner {
            TypeInner::Struct { members, .. } => {
                let widths = members
                    .iter()
                    .map(|m| {
                        if matches!(m.binding, Some(naga::Binding::Location { .. })) {
                            width(&module.types[m.ty].inner)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if widths.iter().all(|w| w.is_none_or(|w| w == 4)) {
                    continue;
                }
                let mut widened = members.clone();
                for (index, member) in widened.iter_mut().enumerate() {
                    member.offset = index as u32 * 16;
                    if widths[index].is_some_and(|w| w < 4) {
                        member.ty = vec4;
                    }
                }
                let span = widened.len() as u32 * 16;
                (
                    module.types.insert(
                        naga::Type {
                            name: None,
                            inner: TypeInner::Struct {
                                members: widened,
                                span,
                            },
                        },
                        naga::Span::UNDEFINED,
                    ),
                    Some(widths),
                )
            }
            ty if matches!(result.binding, Some(naga::Binding::Location { .. })) => {
                let Some(w) = width(ty) else {
                    continue;
                };
                if w == 4 {
                    continue;
                }
                (vec4, Some(vec![Some(w)]))
            }
            _ => continue,
        };
        let structure = matches!(module.types[result.ty].inner, TypeInner::Struct { .. });
        let widths = members.unwrap();
        result.ty = new_type;
        let zero = entry.function.expressions.append(
            Expression::Literal(naga::Literal::F32(0.0)),
            naga::Span::UNDEFINED,
        );
        let one = entry.function.expressions.append(
            Expression::Literal(naga::Literal::F32(1.0)),
            naga::Span::UNDEFINED,
        );
        widen_returns(
            &mut entry.function.expressions,
            &mut entry.function.body,
            new_type,
            vec4,
            &widths,
            structure,
            zero,
            one,
        );
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn widen_returns(
    expressions: &mut naga::Arena<Expression>,
    block: &mut naga::Block,
    result_type: Handle<naga::Type>,
    vec4: Handle<naga::Type>,
    widths: &[Option<u32>],
    structure: bool,
    zero: Handle<Expression>,
    one: Handle<Expression>,
) {
    for statement in block.iter_mut() {
        match statement {
            Statement::Block(b) => widen_returns(
                expressions,
                b,
                result_type,
                vec4,
                widths,
                structure,
                zero,
                one,
            ),
            Statement::If { accept, reject, .. } => {
                widen_returns(
                    expressions,
                    accept,
                    result_type,
                    vec4,
                    widths,
                    structure,
                    zero,
                    one,
                );
                widen_returns(
                    expressions,
                    reject,
                    result_type,
                    vec4,
                    widths,
                    structure,
                    zero,
                    one,
                );
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    widen_returns(
                        expressions,
                        &mut c.body,
                        result_type,
                        vec4,
                        widths,
                        structure,
                        zero,
                        one,
                    );
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                widen_returns(
                    expressions,
                    body,
                    result_type,
                    vec4,
                    widths,
                    structure,
                    zero,
                    one,
                );
                widen_returns(
                    expressions,
                    continuing,
                    result_type,
                    vec4,
                    widths,
                    structure,
                    zero,
                    one,
                );
            }
            Statement::Return { value: Some(value) } => {
                let start = expressions.len();
                let mut members = Vec::new();
                for (index, &width) in widths.iter().enumerate() {
                    let mut component = if structure {
                        expressions.append(
                            Expression::AccessIndex {
                                base: *value,
                                index: index as u32,
                            },
                            naga::Span::UNDEFINED,
                        )
                    } else {
                        *value
                    };
                    if let Some(width) = width.filter(|w| *w < 4) {
                        let mut components = vec![component];
                        for _ in width..3 {
                            components.push(zero);
                        }
                        components.push(one);
                        component = expressions.append(
                            Expression::Compose {
                                ty: vec4,
                                components,
                            },
                            naga::Span::UNDEFINED,
                        );
                    }
                    members.push(component);
                }
                let value = if structure {
                    expressions.append(
                        Expression::Compose {
                            ty: result_type,
                            components: members,
                        },
                        naga::Span::UNDEFINED,
                    )
                } else {
                    members[0]
                };
                let mut replacement = naga::Block::new();
                replacement.push(
                    Statement::Emit(expressions.range_from(start)),
                    naga::Span::UNDEFINED,
                );
                replacement.push(
                    Statement::Return { value: Some(value) },
                    naga::Span::UNDEFINED,
                );
                *statement = Statement::Block(replacement);
            }
            _ => {}
        }
    }
}

// With opaque parameters specialized above, every sampler use names its bound
// global. Infer comparison types before validation, including uses in helpers.
pub(crate) fn resolve_sampler_types(module: &mut naga::Module) -> Result<(), vk::Result> {
    let mut uses = HashMap::<Handle<GlobalVariable>, u8>::new();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expression) in function.expressions.iter() {
            if let Expression::ImageSample {
                sampler, depth_ref, ..
            } = expression
            {
                *uses
                    .entry(global(&function.expressions, *sampler)?)
                    .or_default() |= if depth_ref.is_some() { 2 } else { 1 };
            }
        }
    }
    if uses.values().any(|&kind| kind == 3) {
        return Err(UNSUPPORTED);
    }
    let comparison = module.types.insert(
        naga::Type {
            name: None,
            inner: TypeInner::Sampler { comparison: true },
        },
        naga::Span::UNDEFINED,
    );
    for (global, kind) in uses {
        if kind == 2 {
            module.global_variables[global].ty = comparison;
        }
    }
    Ok(())
}
