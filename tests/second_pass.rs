// A table that stopped part-way on a name another table defines is completed by loading it again once the others are
// loaded (`Interpreter::load_table_again`), keeping what its first pass made.

mod common;

use acpi::aml::AmlError;
use common::*;

/// `Device(TPD0) { Name(FRST, 5) Alias(FRST, ALS1) Name(SECN, Zero) Store(IC1S, SECN) Name(THRD, 9) }`: `IC1S` is
/// defined by [`later`], so a first load stops at the `Store`.
fn needs_later() -> Vec<u8> {
    let body = cat(&[
        b"TPD0",
        &[0x08],
        b"FRST",
        &byte(5),
        &[0x06],
        b"FRST",
        b"ALS1",
        &[0x08],
        b"SECN",
        &[0x00],
        &[0x70],
        b"IC1S",
        b"SECN",
        &[0x08],
        b"THRD",
        &byte(9),
    ]);
    cat(&[&[0x5b, 0x82], &pkg(&body)])
}

/// `Name(IC1S, 42) Method(BUMP) { Store(7, \TPD0.FRST) }`.
fn later() -> Vec<u8> {
    let bump = cat(&[&[0x70], &byte(7), &[b'\\', 0x2e], b"TPD0", b"FRST"]);
    cat(&[&[0x08], b"IC1S", &byte(42), &method(b"BUMP", 0, &bump)])
}

fn integer(interpreter: &acpi::aml::Interpreter<TestHandler>, path: &str) -> Result<u64, AmlError> {
    interpreter.evaluate(name(path), vec![]).and_then(|value| value.as_integer())
}

#[test]
fn a_table_that_stopped_on_a_later_tables_name_is_completed_by_a_second_pass() {
    let (interpreter, _) = interpreter();
    assert!(matches!(interpreter.load_table(&needs_later()), Err(AmlError::ObjectDoesNotExist(_))));
    // The first pass made what came before the stop, and nothing after it.
    assert_eq!(integer(&interpreter, "\\TPD0.FRST"), Ok(5));
    assert!(matches!(integer(&interpreter, "\\TPD0.THRD"), Err(AmlError::ObjectDoesNotExist(_))));

    interpreter.load_table(&later()).unwrap();
    // Something changes an object the first pass made before the second pass runs.
    interpreter.evaluate(name("\\BUMP"), vec![]).unwrap();
    assert_eq!(integer(&interpreter, "\\TPD0.FRST"), Ok(7));

    interpreter.load_table_again(&needs_later()).unwrap();
    assert_eq!(integer(&interpreter, "\\TPD0.FRST"), Ok(7), "the first pass's object is kept, not reset to 5");
    assert_eq!(integer(&interpreter, "\\TPD0.ALS1"), Ok(7), "the existing alias is no collision");
    assert_eq!(integer(&interpreter, "\\TPD0.SECN"), Ok(42), "the code past the stop ran");
    assert_eq!(integer(&interpreter, "\\TPD0.THRD"), Ok(9), "and the objects after it exist");
}

#[test]
fn a_second_pass_that_still_misses_the_name_stops_again_and_changes_nothing_made() {
    let (interpreter, _) = interpreter();
    assert!(interpreter.load_table(&needs_later()).is_err());
    assert!(matches!(interpreter.load_table_again(&needs_later()), Err(AmlError::ObjectDoesNotExist(_))));
    assert_eq!(integer(&interpreter, "\\TPD0.FRST"), Ok(5));
    assert!(matches!(integer(&interpreter, "\\TPD0.THRD"), Err(AmlError::ObjectDoesNotExist(_))));
}

#[test]
fn an_ordinary_load_after_a_second_pass_replaces_objects_as_before() {
    let (interpreter, _) = interpreter();
    let _ = interpreter.load_table(&needs_later());
    let _ = interpreter.load_table_again(&needs_later());
    // `Device(TPD0) { Name(FRST, 1) }`: the keep-existing mode ended with the second pass.
    let again = cat(&[&[0x5b, 0x82], &pkg(&cat(&[b"TPD0", &[0x08], b"FRST", &byte(1)]))]);
    interpreter.load_table(&again).unwrap();
    assert_eq!(integer(&interpreter, "\\TPD0.FRST"), Ok(1));
}
