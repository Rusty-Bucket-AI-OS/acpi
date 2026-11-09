// Hardening: the hardened interpreter: firmware AML that is broken, hostile or uses what is not
// implemented gets an error, never a panic or a hang, and the interpreter stays usable afterwards.

mod common;

use acpi::aml::{
    AmlError,
    Limit,
    Limits,
    object::Object,
    op_region::RegionSpace,
    resource::{InterruptPolarity, InterruptTrigger, Resource, resource_descriptor_list},
};
use common::*;

/// `Method(TST0) { Return(Add(5, 7)) }`.
fn tst0() -> Vec<u8> {
    method(b"TST0", 0, &cat(&[&[0xa4, 0x72], &byte(5), &byte(7), &[0x00]]))
}

fn integer(interpreter: &acpi::aml::Interpreter<TestHandler>, path: &str) -> Result<u64, AmlError> {
    interpreter.evaluate(name(path), vec![]).and_then(|value| value.as_integer())
}

#[test]
fn a_method_returns_a_computed_integer() {
    let (interpreter, _) = interpreter();
    interpreter.load_table(&tst0()).unwrap();
    assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
}

#[test]
fn load_load_table_and_match_are_errors_not_todo() {
    for (body, what) in [(vec![0x5b, 0x20, 0x00], "Load"), (vec![0x5b, 0x1f], "LoadTable"), (vec![0x89], "Match")]
    {
        let (interpreter, _) = interpreter();
        interpreter.load_table(&cat(&[&method(b"UNSP", 0, &body), &tst0()])).unwrap();
        assert_eq!(integer(&interpreter, "\\UNSP"), Err(AmlError::Unsupported(what)));
        // And the interpreter goes on.
        assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
    }
}

#[test]
fn a_stream_cut_mid_opcode_is_an_error() {
    for stream in [&[0x14][..], &[0x5b], &[0x08, b'A', b'B'], &[0x14, 0x05, b'T', b'S'], &[0x0d, b'a', b'b']] {
        let (interpreter, _) = interpreter();
        assert!(interpreter.load_table(stream).is_err(), "{stream:x?}");
    }
}

#[test]
fn a_pkg_length_past_the_end_is_malformed() {
    let (interpreter, _) = interpreter();
    // A method that claims 63 bytes and has 6.
    assert_eq!(
        interpreter.load_table(&[0x14, 0x3f, b'T', b'S', b'T', b'1', 0x00]),
        Err(AmlError::MalformedStream)
    );
    // A scope that claims more than there is, and one whose length is shorter than its own name.
    assert!(interpreter.load_table(&[0x10, 0x30, b'_', b'S', b'B', b'_']).is_err());
    assert!(interpreter.load_table(&[0x10, 0x02, b'_', b'S', b'B', b'_']).is_err());
}

#[test]
fn a_huge_buffer_is_refused_before_it_is_allocated() {
    let (interpreter, _) = small_interpreter();
    let buffer = cat(&[&[0x11], &pkg(&dword(0xffff_ffff))]);
    interpreter.load_table(&method(b"BIGB", 0, &cat(&[&[0xa4], &buffer]))).unwrap();
    assert_eq!(integer(&interpreter, "\\BIGB"), Err(AmlError::LimitExceeded(Limit::BufferSize)));
}

#[test]
fn an_endless_while_is_stopped_by_the_loop_limit_and_by_the_op_budget() {
    let endless = method(b"LOOP", 0, &cat(&[&[0xa2], &pkg(&[0x01])]));
    let (interpreter, _) = small_interpreter();
    interpreter.load_table(&cat(&[&endless, &tst0()])).unwrap();
    assert_eq!(integer(&interpreter, "\\LOOP"), Err(AmlError::LimitExceeded(Limit::LoopIterations)));
    // With the loop limit out of the way, the op budget stops it.
    interpreter.set_limits(Limits { max_ops: 10_000, max_loop_iterations: u64::MAX, ..Limits::default() });
    assert_eq!(integer(&interpreter, "\\LOOP"), Err(AmlError::LimitExceeded(Limit::Ops)));
    // Evaluation still works after a limit was hit.
    assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
}

#[test]
fn a_method_that_calls_itself_runs_into_the_depth_limit_and_leaves_the_stack_clean() {
    let (interpreter, _) = interpreter();
    // `Method(RECU) { RECU() }`, and `Method(CALL) { Return(TST0()) }` to see the context stack afterwards.
    let recurse = method(b"RECU", 0, b"RECU");
    let call = method(b"CALL", 0, &cat(&[&[0xa4], b"TST0"]));
    interpreter.load_table(&cat(&[&recurse, &tst0(), &call])).unwrap();
    assert_eq!(integer(&interpreter, "\\RECU"), Err(AmlError::LimitExceeded(Limit::MethodDepth)));
    // A stale context left behind would have been popped by this `Return`.
    assert_eq!(integer(&interpreter, "\\CALL"), Ok(12));
}

#[test]
fn ops_left_without_operands_in_a_loop_are_bounded() {
    // `While (One) { Add(` : every iteration leaves an `Add` waiting for its operands.
    let (interpreter, _) = interpreter();
    interpreter.set_limits(Limits { max_in_flight: 100, ..Limits::default() });
    let looped = method(b"PILE", 0, &cat(&[&[0xa2], &pkg(&[0x01, 0x72])]));
    interpreter.load_table(&looped).unwrap();
    assert_eq!(integer(&interpreter, "\\PILE"), Err(AmlError::LimitExceeded(Limit::InFlight)));
}

#[test]
fn many_large_buffers_in_one_call_run_into_the_total_budget() {
    // Each 32 KiB buffer is within the single-object limit; a loop making them is not, in all.
    let (interpreter, _) = interpreter();
    interpreter.set_limits(Limits { max_total_bytes: 1024 * 1024, ..Limits::default() });
    let buffer = cat(&[&[0x11], &pkg(&dword(32 * 1024))]);
    let body = cat(&[&[0xa2], &pkg(&cat(&[&[0x01, 0x70], &buffer, &[0x60]]))]);
    interpreter.load_table(&method(b"BUFS", 0, &body)).unwrap();
    assert_eq!(integer(&interpreter, "\\BUFS"), Err(AmlError::LimitExceeded(Limit::TotalBytes)));
}

#[test]
fn a_package_with_too_many_elements_is_refused() {
    let (interpreter, _) = small_interpreter();
    let var = cat(&[&[0x13], &pkg(&[0x0b, 0xff, 0xff])]);
    let fixed = cat(&[&[0x12], &pkg(&[0xff])]);
    interpreter.load_table(&method(b"VPKG", 0, &cat(&[&[0xa4], &var]))).unwrap();
    interpreter.load_table(&method(b"FPKG", 0, &cat(&[&[0xa4], &fixed]))).unwrap();
    assert_eq!(integer(&interpreter, "\\VPKG"), Err(AmlError::LimitExceeded(Limit::PackageElements)));
    assert_eq!(integer(&interpreter, "\\FPKG"), Err(AmlError::LimitExceeded(Limit::PackageElements)));
}

#[test]
fn a_var_package_shorter_than_its_count_completes_with_uninitialized_elements() {
    // Upstream never completed it (the count was mis-computed) and looped forever.
    let (interpreter, _) = interpreter();
    let var = cat(&[&[0x13], &pkg(&cat(&[&byte(3), &byte(9)]))]);
    interpreter.load_table(&method(b"VARP", 0, &cat(&[&[0xa4], &var]))).unwrap();
    let value = interpreter.evaluate(name("\\VARP"), vec![]).unwrap();
    let Object::Package(ref elements) = *value else { panic!("not a package") };
    assert_eq!(elements.len(), 3);
    assert_eq!(elements[0].as_integer(), Ok(9));
}

/// `OperationRegion(R0__, SystemMemory, 0, 4)` and `Field(R0__, ByteAcc) { F0__, 64 }`.
#[test]
fn a_field_past_its_region_is_an_error() {
    let (interpreter, _) = interpreter();
    let region = cat(&[&[0x5b, 0x80], b"R0__", &[0x00, 0x00], &byte(4)]);
    let field = cat(&[&[0x5b, 0x81], &pkg(&cat(&[b"R0__", &[0x01], b"F0__", &bits(64)]))]);
    let read = method(b"RDF0", 0, &cat(&[&[0xa4], b"F0__"]));
    interpreter.load_table(&cat(&[&region, &field, &read])).unwrap();
    assert_eq!(integer(&interpreter, "\\RDF0"), Err(AmlError::RegionAccessOutOfBounds));
}

#[test]
fn a_second_handler_for_a_space_is_an_error_not_an_assert() {
    let (interpreter, _) = interpreter();
    interpreter.install_region_handler(RegionSpace::EmbeddedControl, FakeEc::new()).unwrap();
    assert_eq!(
        interpreter.install_region_handler(RegionSpace::EmbeddedControl, FakeEc::new()),
        Err(AmlError::RegionHandlerAlreadyInstalled(RegionSpace::EmbeddedControl))
    );
}

/// `\_SB.EC0` with an EC region, a 16-bit `ByteAcc` field at 0x10, `_REG`, and methods to read and write it.
fn ec_table() -> Vec<u8> {
    let region = cat(&[&[0x5b, 0x80], b"ECOR", &[0x03, 0x00], &[0x0b, 0x00, 0x01]]);
    let field = cat(&[&[0x5b, 0x81], &pkg(&cat(&[b"ECOR", &[0x01], &[0x00], &bits(0x80), b"B0W_", &bits(16)]))]);
    let ecok = cat(&[&[0x08], b"ECOK", &[0x00]]);
    let reg = method(b"_REG", 2, &cat(&[&[0x70, 0x69], b"ECOK"]));
    let rd = method(b"RD__", 0, &cat(&[&[0xa4], b"B0W_"]));
    let wr = method(b"WR__", 1, &cat(&[&[0x70, 0x68], b"B0W_"]));
    let body = cat(&[&[0x5c, 0x2e], b"_SB_", b"EC0_", &ecok, &region, &field, &reg, &rd, &wr]);
    cat(&[&[0x5b, 0x82], &pkg(&body)])
}

#[test]
fn an_ec_field_goes_through_the_region_handler_byte_by_byte() {
    let (interpreter, _) = interpreter();
    let ec = FakeEc::new();
    interpreter.install_region_handler(RegionSpace::EmbeddedControl, ec.clone()).unwrap();
    interpreter.load_table(&ec_table()).unwrap();
    {
        let mut space = ec.space.lock().unwrap();
        space[0x10] = 0xcd;
        space[0x11] = 0xab;
    }
    assert_eq!(integer(&interpreter, "\\_SB.EC0.RD__"), Ok(0xabcd));
    assert_eq!(*ec.accesses.lock().unwrap(), [(0x10, 8, None), (0x11, 8, None)]);

    ec.accesses.lock().unwrap().clear();
    interpreter.evaluate(name("\\_SB.EC0.WR__"), vec![Object::Integer(0x1234).wrap()]).unwrap();
    assert_eq!(*ec.accesses.lock().unwrap(), [(0x10, 8, Some(0x34)), (0x11, 8, Some(0x12))]);
    assert_eq!(ec.space.lock().unwrap()[0x10..0x12], [0x34, 0x12]);
}

#[test]
fn initialization_runs_reg_for_the_ec_and_counts_failures_without_stopping() {
    let (interpreter, _) = interpreter();
    interpreter.install_region_handler(RegionSpace::EmbeddedControl, FakeEc::new()).unwrap();
    // A device whose `_STA` fails (it returns a string) beside the EC.
    let bad_sta = method(b"_STA", 0, &[0xa4, 0x0d, b'x', 0x00]);
    let bad = cat(&[&[0x5b, 0x82], &pkg(&cat(&[&[0x5c, 0x2e], b"_SB_", b"BAD_", &bad_sta]))]);
    interpreter.load_table(&cat(&[&bad, &ec_table()])).unwrap();
    let report = interpreter.initialize_namespace();
    assert_eq!(report.reg_calls, 1);
    assert_eq!(report.sta_failures, 1);
    assert!(report.devices_initialized >= 1);
    assert_eq!(integer(&interpreter, "\\_SB.EC0.ECOK"), Ok(1));
}

#[test]
fn notify_reaches_the_host_with_the_resolved_path() {
    let (interpreter, handler) = interpreter();
    let device = cat(&[&[0x5b, 0x82], &pkg(&cat(&[&[0x5c, 0x2e], b"_SB_", b"BAT0"]))]);
    let notify = method(b"NTFY", 0, &cat(&[&[0x86, 0x5c, 0x2e], b"_SB_", b"BAT0", &byte(0x80)]));
    interpreter.load_table(&cat(&[&device, &notify])).unwrap();
    interpreter.evaluate(name("\\NTFY"), vec![]).unwrap();
    // The segments keep their padding, as the namespace spells them.
    assert_eq!(handler.0.lock().unwrap().notifies, [("\\_SB_.BAT0".to_string(), 0x80)]);
}

#[test]
fn fatal_is_reported_to_the_host_and_ends_the_evaluation_with_an_error() {
    let (interpreter, handler) = interpreter();
    interpreter.load_table(&method(b"FATL", 0, &cat(&[&[0x5b, 0x32, 1, 2, 0, 0, 0], &byte(3)]))).unwrap();
    assert_eq!(integer(&interpreter, "\\FATL"), Err(AmlError::FatalOpcode { typ: 1, code: 2, arg: 3 }));
    assert_eq!(handler.0.lock().unwrap().fatals, [(1, 2, 3)]);
}

#[test]
fn divide_and_mod_by_zero_are_errors() {
    let (interpreter, _) = interpreter();
    interpreter.load_table(&method(b"DIV0", 0, &[0xa4, 0x78, 0x01, 0x00, 0x00, 0x00])).unwrap();
    interpreter.load_table(&method(b"MOD0", 0, &[0xa4, 0x85, 0x01, 0x00, 0x00])).unwrap();
    assert_eq!(integer(&interpreter, "\\DIV0"), Err(AmlError::DivideByZero));
    assert_eq!(integer(&interpreter, "\\MOD0"), Err(AmlError::DivideByZero));
}

#[test]
fn a_store_through_index_fills_a_package_element() {
    // What `_BST` and `_BIX` methods do to build their answer (it was `todo!()`).
    let (interpreter, _) = interpreter();
    let package = cat(&[&[0x08], b"PKG_", &[0x12], &pkg(&[0x02, 0x00, 0x00])]);
    let body =
        cat(&[&[0x70], &byte(7), &[0x88], b"PKG_", &[0x01, 0x00], &[0xa4, 0x83, 0x88], b"PKG_", &[0x01, 0x00]]);
    interpreter.load_table(&cat(&[&package, &method(b"SETP", 0, &body)])).unwrap();
    assert_eq!(integer(&interpreter, "\\SETP"), Ok(7));
}

#[test]
fn mid_past_the_end_of_a_string_is_the_rest_of_it() {
    // `Mid("ABCD", 1, 10)`: the end used to be past the string (a panic).
    let (interpreter, _) = interpreter();
    let body = cat(&[&[0xa4, 0x9e, 0x0d], b"ABCD", &[0x00, 0x01], &byte(10), &[0x00]]);
    interpreter.load_table(&method(b"MIDS", 0, &body)).unwrap();
    let value = interpreter.evaluate(name("\\MIDS"), vec![]).unwrap();
    assert_eq!(value.as_string().unwrap(), "BCD");
}

#[test]
fn random_streams_never_panic() {
    let seed = std::env::var("AML_TEST_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x5eed_2026_1104u64);
    println!("seed {seed:#x} (AML_TEST_SEED)");
    let mut rng = Rng(seed | 1);
    for _ in 0..2_000 {
        let len = 1 + (rng.next() % 64) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let (interpreter, _) = small_interpreter();
        let _ = interpreter.load_table(&bytes);
        // The same bytes as a method's body, run.
        let (interpreter, _) = small_interpreter();
        if interpreter.load_table(&method(b"RAND", 3, &bytes)).is_ok() {
            let args = vec![Object::Integer(1).wrap(), Object::Integer(2).wrap(), Object::Integer(3).wrap()];
            let _ = interpreter.evaluate(name("\\RAND"), args);
            let _ = interpreter.initialize_namespace();
        }
    }
}

fn gpio_int(pin: u16, source: &str) -> Vec<u8> {
    let name_offset = 23 + 2;
    let vendor_offset = name_offset + source.len() + 1;
    let mut d = vec![0x8c, 0, 0, 1, 0];
    d.extend_from_slice(&1u16.to_le_bytes()); // consumer
    d.extend_from_slice(&0x0002u16.to_le_bytes()); // level, active low
    d.push(0); // default pull
    d.extend_from_slice(&0u16.to_le_bytes()); // drive strength
    d.extend_from_slice(&500u16.to_le_bytes()); // debounce
    d.extend_from_slice(&23u16.to_le_bytes()); // pin table
    d.push(0);
    d.extend_from_slice(&(name_offset as u16).to_le_bytes());
    d.extend_from_slice(&(vendor_offset as u16).to_le_bytes());
    d.extend_from_slice(&0u16.to_le_bytes());
    d.extend_from_slice(&pin.to_le_bytes());
    d.extend_from_slice(source.as_bytes());
    d.push(0);
    let len = (d.len() - 3) as u16;
    d[1..3].copy_from_slice(&len.to_le_bytes());
    d
}

fn i2c(address: u16, speed: u32, source: &str) -> Vec<u8> {
    let mut d = vec![0x8e, 0, 0, 1, 0, 1, 0x02];
    d.extend_from_slice(&0u16.to_le_bytes());
    d.push(1);
    d.extend_from_slice(&6u16.to_le_bytes());
    d.extend_from_slice(&speed.to_le_bytes());
    d.extend_from_slice(&address.to_le_bytes());
    d.extend_from_slice(source.as_bytes());
    d.push(0);
    let len = (d.len() - 3) as u16;
    d[1..3].copy_from_slice(&len.to_le_bytes());
    d
}

#[test]
fn a_touchpads_crs_decodes_its_i2c_address_and_gpio_interrupt() {
    let crs = cat(&[&i2c(0x2c, 400_000, "\\_SB.PCI0.I2C1"), &gpio_int(27, "\\_SB.GPO0"), &[0x79, 0x00]]);
    let resources = resource_descriptor_list(Object::Buffer(crs).wrap()).unwrap();
    let [Resource::SerialBus(bus), Resource::Gpio(gpio)] = &resources[..] else { panic!("{resources:?}") };
    assert_eq!((bus.bus_type, bus.slave_address, bus.connection_speed), (1, 0x2c, 400_000));
    assert_eq!(bus.resource_source, "\\_SB.PCI0.I2C1");
    assert!(gpio.is_interrupt && gpio.is_consumer);
    assert_eq!((gpio.trigger, gpio.polarity), (InterruptTrigger::Level, InterruptPolarity::ActiveLow));
    assert_eq!(gpio.pins, [27]);
    assert_eq!(gpio.resource_source, "\\_SB.GPO0");
}

#[test]
fn truncated_or_unknown_resource_descriptors_are_errors_or_skipped_never_panics() {
    for bad in
        [&[0x8c, 0xff, 0x00][..], &[0x86], &[0x8c, 0x02, 0x00, 1, 0], &[0x47, 1], &[0x89, 0x04, 0x00, 1, 1, 0, 0]]
    {
        assert!(resource_descriptor_list(Object::Buffer(bad.to_vec()).wrap()).is_err(), "{bad:x?}");
    }
    // A vendor-defined small descriptor (once `unimplemented!()`) is skipped and the IRQ after it read.
    let list = [&[0x71, 0xaa][..], &[0x22, 0x02, 0x00], &[0x79, 0x00]].concat();
    let resources = resource_descriptor_list(Object::Buffer(list).wrap()).unwrap();
    assert!(matches!(
        resources[..],
        [Resource::Unsupported { large: false, descriptor_type: 0x0e }, Resource::Irq(_)]
    ));
}

#[test]
fn random_opcode_soup_never_panics() {
    // Streams built from real AML pieces (opcodes, names, integers, lengths), so the interpreter gets past the
    // first byte far more often than with plain random bytes.
    let seed = std::env::var("AML_TEST_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0xa31_2026_1104u64);
    println!("seed {seed:#x} (AML_TEST_SEED)");
    let pieces: &[&[u8]] = &[
        &[0x00],
        &[0x01],
        &[0xff],
        &[0x0a, 0x05],
        &[0x0b, 0x00, 0x10],
        &[0x0c, 0xff, 0xff, 0xff, 0xff],
        &[0x0d, b'A', 0x00],
        b"ABCD",
        b"_SB_",
        &[0x5c],
        &[0x5e],
        &[0x2e],
        &[0x60],
        &[0x61],
        &[0x68],
        &[0x69],
        &[0x70],
        &[0x71],
        &[0x72],
        &[0x73],
        &[0x74],
        &[0x75],
        &[0x76],
        &[0x77],
        &[0x78],
        &[0x79],
        &[0x7a],
        &[0x7b],
        &[0x80],
        &[0x83],
        &[0x84],
        &[0x85],
        &[0x86],
        &[0x87],
        &[0x88],
        &[0x8a],
        &[0x8e],
        &[0x90],
        &[0x92],
        &[0x93],
        &[0x96],
        &[0x97],
        &[0x98],
        &[0x99],
        &[0x9c],
        &[0x9d],
        &[0x9e],
        &[0x9f],
        &[0xa0],
        &[0xa1],
        &[0xa2],
        &[0xa3],
        &[0xa4],
        &[0xa5],
        &[0x08],
        &[0x10],
        &[0x11],
        &[0x12],
        &[0x13],
        &[0x14],
        &[0x5b, 0x01],
        &[0x5b, 0x02],
        &[0x5b, 0x12],
        &[0x5b, 0x13],
        &[0x5b, 0x21],
        &[0x5b, 0x22],
        &[0x5b, 0x23],
        &[0x5b, 0x24],
        &[0x5b, 0x25],
        &[0x5b, 0x26],
        &[0x5b, 0x27],
        &[0x5b, 0x28],
        &[0x5b, 0x29],
        &[0x5b, 0x31],
        &[0x5b, 0x33],
        &[0x5b, 0x80],
        &[0x5b, 0x81],
        &[0x5b, 0x82],
        &[0x5b, 0x83],
        &[0x5b, 0x84],
        &[0x5b, 0x85],
        &[0x5b, 0x86],
        &[0x5b, 0x87],
        &[0x05],
        &[0x10],
        &[0x3f],
        &[0x41, 0x01],
        &[0x02],
        &[0x03],
    ];
    let mut rng = Rng(seed | 1);
    for _ in 0..3_000 {
        let count = 1 + (rng.next() % 40) as usize;
        let mut bytes = Vec::new();
        for _ in 0..count {
            bytes.extend_from_slice(pieces[(rng.next() % pieces.len() as u64) as usize]);
        }
        let (interpreter, _) = small_interpreter();
        interpreter.install_region_handler(RegionSpace::EmbeddedControl, FakeEc::new()).unwrap();
        let _ = interpreter.load_table(&bytes);
        let _ = interpreter.load_table(&method(b"SOUP", 2, &bytes));
        let _ = interpreter.evaluate(name("\\SOUP"), vec![Object::Integer(3).wrap(), Object::Integer(0).wrap()]);
        let _ = interpreter.initialize_namespace();
        // The interpreter still works.
        interpreter.load_table(&tst0()).unwrap();
        assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
    }
}

/// `Name(BUFF, Buffer(4) {})` inside a method.
fn four_byte_buffer() -> Vec<u8> {
    cat(&[&[0x08], b"BUFF", &[0x11], &pkg(&byte(4))])
}

#[test]
fn a_huge_create_field_is_refused_when_it_is_created() {
    // `CreateField(BUFF, 0, 0xFFFFFFFF, FLD0)`: 4 Gbit over a 4-byte buffer (a 512 MiB allocation when read, before).
    let (interpreter, _) = interpreter();
    let create = cat(&[&[0x5b, 0x13], b"BUFF", &[0x00], &dword(0xffff_ffff), b"FLD0"]);
    let body = cat(&[&four_byte_buffer(), &create, &[0xa4], b"FLD0"]);
    interpreter.load_table(&cat(&[&method(b"HUGE", 0, &body), &tst0()])).unwrap();
    assert_eq!(integer(&interpreter, "\\HUGE"), Err(AmlError::IndexOutOfBounds));
    assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
}

#[test]
fn a_fixed_size_field_past_its_buffer_is_refused_and_one_inside_works() {
    // `CreateDWordField(BUFF, 2, FLD1)` over 4 bytes ends at bit 48; `CreateWordField(BUFF, 2, FLD2)` fits.
    let (interpreter, _) = interpreter();
    let past = cat(&[&four_byte_buffer(), &[0x8a], b"BUFF", &byte(2), b"FLD1", &[0xa4], b"FLD1"]);
    let inside = cat(&[&four_byte_buffer(), &[0x8b], b"BUFF", &byte(2), b"FLD2", &[0xa4], b"FLD2"]);
    interpreter.load_table(&cat(&[&method(b"PAST", 0, &past), &method(b"INSD", 0, &inside)])).unwrap();
    assert_eq!(integer(&interpreter, "\\PAST"), Err(AmlError::IndexOutOfBounds));
    // (The method returns the field object itself; creating it succeeded.)
    assert!(interpreter.evaluate(name("\\INSD"), vec![]).is_ok());
    // A field of no bits.
    let empty = cat(&[&four_byte_buffer(), &[0x5b, 0x13], b"BUFF", &[0x00, 0x00], b"FLD3", &[0xa4], b"FLD3"]);
    interpreter.load_table(&method(b"NONE", 0, &empty)).unwrap();
    assert_eq!(integer(&interpreter, "\\NONE"), Err(AmlError::IndexOutOfBounds));
}

#[test]
fn a_loop_of_sleeps_is_ended_by_the_call_deadline() {
    // `While (One) { Sleep (1000) }`: it held the AML lock for hours; now the call ends after `max_call_ms`.
    let (interpreter, handler) = interpreter();
    let body = cat(&[&[0xa2], &pkg(&[0x01, 0x5b, 0x22, 0x0b, 0xe8, 0x03])]);
    interpreter.load_table(&cat(&[&method(b"NAPS", 0, &body), &tst0()])).unwrap();
    assert_eq!(integer(&interpreter, "\\NAPS"), Err(AmlError::LimitExceeded(Limit::Deadline)));
    let slept: u64 = handler.0.lock().unwrap().sleeps.iter().sum();
    assert!((5000..=5001).contains(&slept), "slept {slept} ms");
    // One `Sleep(0xFFFFFFFF)` sleeps only to the deadline, and the next call gets a full budget again.
    let long = cat(&[&[0x5b, 0x22], &dword(0xffff_ffff), &[0xa4], &byte(1)]);
    interpreter.load_table(&method(b"LONG", 0, &long)).unwrap();
    assert_eq!(integer(&interpreter, "\\LONG"), Err(AmlError::LimitExceeded(Limit::Deadline)));
    assert!(*handler.0.lock().unwrap().sleeps.last().unwrap() <= 5001);
    assert_eq!(integer(&interpreter, "\\TST0"), Ok(12));
}
