//! Source-compatibility and ownership checks for the completion boundary.

use core::time::Duration;
use sgfx_core::backend::{
    CommandExecutor, CommandSubmitter, Completion, CompletionStatus, SubmitError,
};
use sgfx_core::ir::{CommandBuffer, CommandEncoder, ResourceTable};

struct LegacyExecutor;

impl CommandExecutor for LegacyExecutor {
    type Error = ();

    fn execute<'r, 'data>(&mut self, _: &CommandBuffer<'r, 'data>) -> Result<(), ()> {
        Ok(())
    }
}

#[derive(Debug)]
struct EmptyReceipt;

impl Completion for EmptyReceipt {
    type Error = ();

    fn poll(&self) -> Result<CompletionStatus, ()> {
        Ok(CompletionStatus::Complete)
    }

    fn wait(&self, _: Option<Duration>) -> Result<CompletionStatus, ()> {
        self.poll()
    }
}

impl CommandSubmitter for LegacyExecutor {
    type Submission = EmptyReceipt;

    fn submit<'r, 'data>(
        &mut self,
        commands: &CommandBuffer<'r, 'data>,
    ) -> Result<EmptyReceipt, SubmitError<(), EmptyReceipt>> {
        // This interface fixture handles only an empty queue checkpoint, not
        // GPU work or the asynchronous Scarlet implementation requirement.
        if commands.commands().is_empty() {
            Ok(EmptyReceipt)
        } else {
            Err(SubmitError::Rejected(()))
        }
    }
}

#[test]
fn existing_executor_boundary_remains_object_safe() {
    let table = ResourceTable::new();
    let commands = CommandEncoder::new(&table).finish().expect("empty stream");
    let mut legacy = LegacyExecutor;
    let executor: &mut dyn CommandExecutor<Error = ()> = &mut legacy;
    assert_eq!(executor.execute(&commands), Ok(()));
}

#[test]
fn receipt_outlives_executor_table_and_commands() {
    fn owned_receipt<S: CommandSubmitter>(executor: &mut S) -> S::Submission
    where
        S::Error: core::fmt::Debug,
        S::Submission: core::fmt::Debug,
    {
        let table = ResourceTable::new();
        let commands = CommandEncoder::new(&table).finish().expect("empty stream");
        executor.submit(&commands).expect("checkpoint")
    }
    let receipt = owned_receipt(&mut LegacyExecutor);
    let erased: Box<dyn Completion<Error = ()>> = Box::new(receipt);
    assert_eq!(erased.poll(), Ok(CompletionStatus::Complete));
    assert_eq!(
        erased.wait(Some(Duration::ZERO)),
        Ok(CompletionStatus::Complete)
    );
    assert_eq!(erased.wait(None), Ok(CompletionStatus::Complete));
}

#[test]
fn failed_submit_retains_a_separately_observable_receipt() {
    let failed = SubmitError::Failed {
        error: "a later chunk was rejected",
        completion: EmptyReceipt,
    };
    let SubmitError::Failed { error, completion } = failed else {
        panic!("expected partial failure");
    };
    assert_eq!(error, "a later chunk was rejected");
    assert_eq!(completion.poll(), Ok(CompletionStatus::Complete));
}

#[test]
fn facade_mapping_preserves_acceptance_classification() {
    let busy: SubmitError<(), EmptyReceipt> = SubmitError::Busy;
    assert!(matches!(
        busy.map(|_| panic!("no error payload"), |_| panic!("no receipt")),
        SubmitError::Busy
    ));
    let rejected: SubmitError<&str, EmptyReceipt> = SubmitError::Rejected("bad descriptor");
    assert!(matches!(
        rejected.map(str::len, |_| panic!("no receipt")),
        SubmitError::Rejected(14)
    ));
    let failed = SubmitError::Failed {
        error: 7,
        completion: EmptyReceipt,
    };
    let mapped = failed.map(|error| error + 1, Box::new);
    let SubmitError::Failed { error, completion } = mapped else {
        panic!("lost failed-prefix state");
    };
    assert_eq!(error, 8);
    assert_eq!(completion.poll(), Ok(CompletionStatus::Complete));
}
