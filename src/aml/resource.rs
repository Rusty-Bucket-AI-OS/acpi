use super::object::WrappedObject;
use crate::aml::{AmlError, Operation, object::Object};
use alloc::{string::String, vec::Vec};
use bit_field::BitField;
use byteorder::{ByteOrder, LittleEndian};
use core::mem;

#[derive(Debug, PartialEq, Eq)]
pub enum Resource {
    Irq(IrqDescriptor),
    AddressSpace(AddressSpaceDescriptor),
    MemoryRange(MemoryRangeDescriptor),
    IOPort(IOPortDescriptor),
    Dma(DMADescriptor),
    // Hardening: the descriptors a laptop's `_CRS` needs for HID over I2C: its interrupt line and its bus address.
    Gpio(GpioDescriptor),
    SerialBus(SerialBusDescriptor),
    /// Hardening: a well-formed descriptor this crate does not decode (it was `unimplemented!()`): skipped, so the rest
    /// of the list is still read.
    Unsupported {
        large: bool,
        descriptor_type: u8,
    },
}

/// Parse a `ResourceDescriptor` buffer into a list of resources.
pub fn resource_descriptor_list(descriptor: WrappedObject) -> Result<Vec<Resource>, AmlError> {
    if let Object::Buffer(ref bytes) = *descriptor {
        let mut descriptors = Vec::new();
        let mut bytes = bytes.as_slice();

        while !bytes.is_empty() {
            let (descriptor, remaining_bytes) = resource_descriptor(bytes)?;

            if let Some(descriptor) = descriptor {
                descriptors.push(descriptor);
                bytes = remaining_bytes;
            } else {
                break;
            }
        }

        Ok(descriptors)
    } else {
        Err(AmlError::InvalidOperationOnObject { op: Operation::ParseResource, typ: descriptor.typ() })
    }
}

/// Hardening: every length is checked against the buffer (the header and `split_at` panicked on a short one).
fn resource_descriptor(bytes: &[u8]) -> Result<(Option<Resource>, &[u8]), AmlError> {
    let Some(&first) = bytes.first() else {
        return Err(AmlError::InvalidResourceDescriptor);
    };
    /*
     * If bit 7 of Byte 0 is set, it's a large descriptor. If not, it's a small descriptor.
     */
    if first.get_bit(7) {
        /*
         * We're parsing a large item. The descriptor type is encoded in Bits 0-6 of Byte 0. Byte 1 contains bits 0-7
         * of the length, and Byte 2 contains bits 8-15 of the length. Subsequent bytes contain the actual data items.
         */
        let descriptor_type = first.get_bits(0..7);
        let length = match bytes.get(1..3) {
            Some(length) => LittleEndian::read_u16(length) as usize,
            None => return Err(AmlError::InvalidResourceDescriptor),
        };
        if bytes.len() < length + 3 {
            return Err(AmlError::InvalidResourceDescriptor);
        }
        let (descriptor_bytes, remaining_bytes) = bytes.split_at(length + 3);

        let descriptor = match descriptor_type {
            0x06 => fixed_memory_descriptor(descriptor_bytes),
            0x07 => address_space_descriptor::<u32>(descriptor_bytes),
            0x08 => address_space_descriptor::<u16>(descriptor_bytes),
            0x09 => extended_interrupt_descriptor(descriptor_bytes),
            0x0a => address_space_descriptor::<u64>(descriptor_bytes),
            0x0c => gpio_connection_descriptor(descriptor_bytes),
            0x0e => serial_bus_descriptor(descriptor_bytes),
            0x01..=0x05 | 0x0b | 0x0d | 0x0f..=0x12 => Ok(Resource::Unsupported { large: true, descriptor_type }),
            _ => Err(AmlError::InvalidResourceDescriptor),
        }?;

        Ok((Some(descriptor), remaining_bytes))
    } else {
        /*
         * We're parsing a small descriptor. Byte 0 has the format:
         *    | Bits        | Field             |
         *    |-------------|-------------------|
         *    | 0-2         | Length - n bytes  |
         *    | 3-6         | Small item type   |
         *    | 7           | 0 = small item    |
         */
        let descriptor_type = first.get_bits(3..=6);
        let length: usize = first.get_bits(0..=2) as usize;
        if bytes.len() < length + 1 {
            return Err(AmlError::InvalidResourceDescriptor);
        }
        let (descriptor_bytes, remaining_bytes) = bytes.split_at(length + 1);

        let descriptor = match descriptor_type {
            0x04 => irq_format_descriptor(descriptor_bytes),
            0x05 => dma_format_descriptor(descriptor_bytes),
            0x08 => io_port_descriptor(descriptor_bytes),
            0x06 | 0x07 | 0x09 | 0x0A | 0x0E => Ok(Resource::Unsupported { large: false, descriptor_type }),
            0x0F => return Ok((None, &[])),
            _ => Err(AmlError::InvalidResourceDescriptor),
        }?;

        Ok((Some(descriptor), remaining_bytes))
    }
}

/// Hardening: a GPIO Connection Descriptor (ACPI 6.5, 6.4.3.8.1): an interrupt (`GpioInt`) or I/O (`GpioIo`) line.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GpioDescriptor {
    /// True for `GpioInt`, false for `GpioIo`.
    pub is_interrupt: bool,
    pub is_consumer: bool,
    /// For `GpioInt`.
    pub trigger: InterruptTrigger,
    /// For `GpioInt`; `ActiveBoth` reads as `ActiveHigh` with `active_both` set.
    pub polarity: InterruptPolarity,
    pub active_both: bool,
    pub is_shared: bool,
    pub is_wake_capable: bool,
    /// 0 default, 1 pull-up, 2 pull-down, 3 no pull, 0x80 and above vendor-defined.
    pub pin_config: u8,
    /// Hundredths of a millisecond.
    pub debounce_timeout: u16,
    /// The pins, as the GPIO controller numbers them.
    pub pins: Vec<u16>,
    /// The GPIO controller's path, e.g. `\_SB.GPO0`.
    pub resource_source: String,
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16, AmlError> {
    bytes.get(at..at + 2).map(LittleEndian::read_u16).ok_or(AmlError::InvalidResourceDescriptor)
}

/// A NUL-terminated string from `start` (to the end of the descriptor if there is no NUL).
fn string_at(bytes: &[u8], start: usize) -> Result<String, AmlError> {
    let tail = bytes.get(start..).ok_or(AmlError::InvalidResourceDescriptor)?;
    let end = tail.iter().position(|b| *b == 0).unwrap_or(tail.len());
    Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn gpio_connection_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * Byte 3 revision, 4 connection type (0 interrupt, 1 I/O), 5-6 general flags (bit 0 consumer), 7-8 interrupt
     * and I/O flags (interrupt: bit 0 edge, bits 2:1 polarity, bit 3 shared, bit 4 wake), 9 pin configuration,
     * 10-11 output drive strength, 12-13 debounce timeout, 14-15 pin table offset, 16 resource source index,
     * 17-18 resource source name offset, 19-20 vendor data offset, 21-22 vendor data length. Offsets count from
     * byte 0.
     */
    if bytes.len() < 23 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let is_interrupt = bytes[4] == 0;
    let general = u16_at(bytes, 5)?;
    let flags = u16_at(bytes, 7)?;
    let pin_table = u16_at(bytes, 14)? as usize;
    let source_name = u16_at(bytes, 17)? as usize;
    if pin_table > source_name || source_name > bytes.len() {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let (pins, _) = bytes[pin_table..source_name].as_chunks::<2>();
    let pins = pins.iter().map(|pin| u16::from_le_bytes(*pin)).collect();
    let polarity_bits = flags.get_bits(1..3);
    Ok(Resource::Gpio(GpioDescriptor {
        is_interrupt,
        is_consumer: general.get_bit(0),
        trigger: if flags.get_bit(0) { InterruptTrigger::Edge } else { InterruptTrigger::Level },
        polarity: if polarity_bits == 1 { InterruptPolarity::ActiveLow } else { InterruptPolarity::ActiveHigh },
        active_both: polarity_bits == 2,
        is_shared: flags.get_bit(3),
        is_wake_capable: flags.get_bit(4),
        pin_config: bytes[9],
        debounce_timeout: u16_at(bytes, 12)?,
        pins,
        resource_source: string_at(bytes, source_name)?,
    }))
}

/// Hardening: a GenericSerialBus Connection Descriptor (ACPI 6.5, 6.4.3.8.2). I2C is decoded; SPI and UART keep their
/// type and source only.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SerialBusDescriptor {
    /// 1 I2C, 2 SPI, 3 UART, 4 CSI-2.
    pub bus_type: u8,
    pub is_consumer: bool,
    /// For I2C: the bus speed in hertz.
    pub connection_speed: u32,
    /// For I2C: the device's address.
    pub slave_address: u16,
    /// For I2C: the address is 10 bits.
    pub ten_bit_address: bool,
    /// The bus controller's path, e.g. `\_SB.PCI0.I2C1`.
    pub resource_source: String,
}

fn serial_bus_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * Byte 3 revision, 4 resource source index, 5 bus type, 6 general flags (bit 1 consumer), 7-8 type-specific
     * flags, 9 type-specific revision, 10-11 type data length (from byte 12), then the type data, then the
     * resource source string. For I2C the type data is the connection speed (12-15) and the address (16-17), and
     * flag bit 0 is 10-bit addressing.
     */
    if bytes.len() < 12 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let bus_type = bytes[5];
    let type_flags = u16_at(bytes, 7)?;
    let type_length = u16_at(bytes, 10)? as usize;
    let source = 12 + type_length;
    if source > bytes.len() {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let (connection_speed, slave_address) = if bus_type == 1 && type_length >= 6 {
        (LittleEndian::read_u32(&bytes[12..16]), u16_at(bytes, 16)?)
    } else {
        (0, 0)
    };
    Ok(Resource::SerialBus(SerialBusDescriptor {
        bus_type,
        is_consumer: bytes[6].get_bit(1),
        connection_speed,
        slave_address,
        ten_bit_address: bus_type == 1 && type_flags.get_bit(0),
        resource_source: string_at(bytes, source)?,
    }))
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InterruptTrigger {
    Edge,
    Level,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InterruptPolarity {
    ActiveHigh,
    ActiveLow,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum AddressSpaceResourceType {
    MemoryRange,
    IORange,
    BusNumberRange,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum AddressSpaceDecodeType {
    Additive,
    Subtractive,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AddressSpaceDescriptor {
    pub resource_type: AddressSpaceResourceType,
    pub is_maximum_address_fixed: bool,
    pub is_minimum_address_fixed: bool,
    pub decode_type: AddressSpaceDecodeType,

    pub granularity: u64,
    pub address_range: (u64, u64),
    pub translation_offset: u64,
    pub length: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryRangeDescriptor {
    FixedLocation { is_writable: bool, base_address: u32, range_length: u32 },
}

fn fixed_memory_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * -- 32-bit Fixed Memory Descriptor ---
     * Offset     Field Name                              Definition
     * Byte 0     32-bit Fixed Memory Range Descriptor    Value = 0x86 (10000110B) – Type = 1, Large item name = 0x06
     * Byte 1     Length, bits [7:0]                      Value = 0x09 (9)
     * Byte 2     Length, bits [15:8]                     Value = 0x00
     * Byte 3     Information                             This field provides extra information about this memory.
     *                                                    Bit [7:1]   Ignored
     *                                                    Bit [0]     Write status, _RW
     *                                                        1  writeable (read/write)
     *                                                        0  non-writeable (read-only)
     * Byte 4     Range base address, _BAS bits [7:0]     Address bits [7:0] of the base memory address for which the card may be configured.
     * Byte 5     Range base address, _BAS bits [15:8]    Address bits [15:8] of the base memory address for which the card may be configured.
     * Byte 6     Range base address, _BAS bits [23:16]   Address bits [23:16] of the base memory address for which the card may be configured.
     * Byte 7     Range base address, _BAS bits [31:24]   Address bits [31:24] of the base memory address for which the card may be configured.
     * Byte 8     Range length, _LEN bits [7:0]           This field contains bits [7:0] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 9     Range length, _LEN bits [15:8]          This field contains bits [15:8] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 10    Range length, _LEN bits [23:16]         This field contains bits [23:16] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 11    Range length, _LEN bits [31:24]         This field contains bits [31:24] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     */
    if bytes.len() != 12 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let information = bytes[3];
    let is_writable = information.get_bit(0);

    let base_address = LittleEndian::read_u32(&bytes[4..=7]);
    let range_length = LittleEndian::read_u32(&bytes[8..=11]);

    Ok(Resource::MemoryRange(MemoryRangeDescriptor::FixedLocation { is_writable, base_address, range_length }))
}

fn address_space_descriptor<T>(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * WORD Address Space Descriptor Definition
     * Note: The definitions for DWORD and QWORD are the same other than the width of the address fields.
     *
     * Offset  Field Name                                   Definition
     * Byte 0  WORD Address Space Descriptor                Value = 0x88 (10001000B) – Type = 1, Large item name = 0x08
     * Byte 1  Length, bits [7:0]                           Variable length, minimum value = 0x0D (13)
     * Byte 2  Length, bits [15:8]                          Variable length, minimum value = 0x00
     * Byte 3  Resource Type                                Indicates which type of resource this descriptor describes. Defined values are:
     *                                                        0         Memory range
     *                                                        1         I/O range
     *                                                        2         Bus number range
     *                                                        3–191     Reserved
     *                                                        192-255   Hardware Vendor Defined
     * Byte 4  General Flags                                Flags that are common to all resource types:
     *                                                        Bits [7:4]   Reserved (must be 0)
     *                                                        Bit [3]     Max Address Fixed, _MAF:
     *                                                          1  The specified maximum address is fixed
     *                                                          0  The specified maximum address is not fixed
     *                                                             and can be changed
     *                                                        Bit [2]      Min Address Fixed,_MIF:
     *                                                          1   The specified minimum address is fixed
     *                                                          0   The specified minimum address is not fixed
     *                                                              and can be changed
     *                                                        Bit [1]      Decode Type, _DEC:
     *                                                          1   This bridge subtractively decodes this address          (top level bridges only)
     *                                                          0   This bridge positively decodes this address
     *                                                        Bit [0]      Ignored
     * Byte 5  Type Specific Flags                           Flags that are specific to each resource type. The meaning of the flags in this field depends on the value of the Resource Type field (see above).
     * Byte 6  Address space granularity, _GRA bits[7:0]     A set bit in this mask means that this bit is decoded. All bits less significant than the most significant set bit must be set. (In other words, the value of the full Address Space Granularity field (all 16 bits) must be a number (2n-1).
     * Byte 7  Address space granularity, _GRA bits[15:8]
     * Byte 8  Address range minimum, _MIN, bits [7:0]       For bridges that translate addresses, this is the address space on the secondary side of the bridge.
     * Byte 9  Address range minimum, _MIN, bits [15:8]
     * Byte 10 Address range maximum, _MAX, bits [7:0]       For bridges that translate addresses, this is the address space on the secondary side of the bridge.
     * Byte 11 Address range maximum, _MAX, bits [15:8]
     * Byte 12 Address Translation offset, _TRA, bits [7:0]  For bridges that translate addresses across the bridge, this is the offset that must be added to the address on the secondary side to obtain the address on the primary side. Non-bridge devices must list 0 for all Address Translation offset bits.
     * Byte 13 Address Translation offset, _TRA, bits [15:8]
     * Byte 14 Address Length, _LEN, bits [7:0]
     * Byte 15 Address Length, _LEN, bits [15:8]
     * Byte 16 Resource Source Index (Optional)              Only present if Resource Source (below) is present. This field gives an index to the specific resource descriptor that this device consumes from in the current resource template for the device object pointed to in Resource Source.
     * String  Resource Source (Optional)                    If present, the device that uses this descriptor consumes its resources from the resources produced by the named device object. If not present, the device consumes its resources out of a global pool. If not present, the device consumes this resource from its hierarchical parent.
     */
    let size = mem::size_of::<T>();

    if bytes.len() < 6 + size * 5 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let resource_type = match bytes[3] {
        0 => AddressSpaceResourceType::MemoryRange,
        1 => AddressSpaceResourceType::IORange,
        2 => AddressSpaceResourceType::BusNumberRange,
        3..=191 => return Err(AmlError::InvalidResourceDescriptor),
        // Hardening: vendor-defined types were `unimplemented!()`.
        192..=255 => {
            return Ok(Resource::Unsupported { large: true, descriptor_type: bytes[0].get_bits(0..7) });
        }
    };

    let general_flags = bytes[4];
    let is_maximum_address_fixed = general_flags.get_bit(3);
    let is_minimum_address_fixed = general_flags.get_bit(2);
    let decode_type = if general_flags.get_bit(1) {
        AddressSpaceDecodeType::Subtractive
    } else {
        AddressSpaceDecodeType::Additive
    };

    // Hardening: the length was checked at the top; the reads are indexed, with no `unwrap`s.
    let field = |n: usize| LittleEndian::read_uint(&bytes[6 + n * size..6 + (n + 1) * size], size);
    let granularity = field(0);
    let address_range_min = field(1);
    let address_range_max = field(2);
    let translation_offset = field(3);
    let length = field(4);

    Ok(Resource::AddressSpace(AddressSpaceDescriptor {
        resource_type,
        is_maximum_address_fixed,
        is_minimum_address_fixed,
        decode_type,
        granularity,
        address_range: (address_range_min, address_range_max),
        translation_offset,
        length,
    }))
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IrqDescriptor {
    pub is_consumer: bool,
    pub trigger: InterruptTrigger,
    pub polarity: InterruptPolarity,
    pub is_shared: bool,
    pub is_wake_capable: bool,
    /*
     * NOTE: We currently only support the cases where a descriptor only contains a single interrupt
     * number.
     */
    pub irq: u32,
}

fn irq_format_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * IRQ Descriptor Definition
     *
     * Offset   Field Name
     * Byte 0   Value = 0x22 or 0x23 (0010001nB)– Type = 0, Small item name = 0x4, Length = 2 or 3
     * Byte 1   IRQ mask bits[7:0], _INT
     *          Bit [0] represents IRQ0, bit[1] is IRQ1, and so on.
     * Byte 2   IRQ mask bits[15:8], _INT
     *          Bit [0] represents IRQ8, bit[1] is IRQ9, and so on.
     * Byte 3   IRQ Information. Each bit, when set, indicates this device is capable of driving a certain type of interrupt.
     *          (Optional—if not included then assume edge sensitive, high true interrupts.)
     *          These bits can be used both for reporting and setting IRQ resources.
     *          Note: This descriptor is meant for describing interrupts that are connected to PIC-compatible interrupt
     *                controllers, which can only be programmed for Active-High-Edge-Triggered or Active-Low-Level-Triggered
     *                interrupts. Any other combination is invalid. The Extended Interrupt Descriptor can be used to describe
     *                other combinations.
     *            Bit [7:6]  Reserved (must be 0)
     *            Bit [5]    Wake Capability, _WKC
     *              0x0 = Not Wake Capable: This interrupt is not capable of waking the system.
     *              0x1 = Wake Capable: This interrupt is capable of waking the system from a
     *                    low-power idle state or a system sleep state.
     *            Bit [4]    Interrupt Sharing, _SHR
     *              0x0 = Exclusive: This interrupt is not shared with other devices.
     *              0x1 = Shared: This interrupt is shared with other devices.
     *            Bit [3]    Interrupt Polarity, _LL
     *              0  Active-High – This interrupt is sampled when the signal is high, or true
     *              1  Active-Low – This interrupt is sampled when the signal is low, or false.
     *            Bit [2:1]  Ignored
     *            Bit [0]    Interrupt Mode, _HE
     *              0  Level-Triggered – Interrupt is triggered in response to signal in a low state.
     *              1  Edge-Triggered – Interrupt is triggered in response to a change in signal state from low to high.
     */

    match bytes.len() {
        0..=2 => Err(AmlError::InvalidResourceDescriptor),
        3 => {
            // no IRQ information ("length 2" in spec)
            let irq = LittleEndian::read_u16(&bytes[1..=2]);

            Ok(Resource::Irq(IrqDescriptor {
                irq: irq as u32,
                is_wake_capable: false,
                is_shared: false,
                polarity: InterruptPolarity::ActiveHigh,
                trigger: InterruptTrigger::Edge,

                is_consumer: false, // assumed to be producer
            }))
        }
        4 => {
            // with IRQ information ("length 3" in spec)
            let irq = LittleEndian::read_u16(&bytes[1..=2]);

            let information = bytes[3];
            let is_wake_capable = information.get_bit(5);
            let is_shared = information.get_bit(4);
            let polarity = match information.get_bit(3) {
                false => InterruptPolarity::ActiveHigh,
                true => InterruptPolarity::ActiveLow,
            };
            let trigger = match information.get_bit(0) {
                false => InterruptTrigger::Level,
                true => InterruptTrigger::Edge,
            };

            Ok(Resource::Irq(IrqDescriptor {
                irq: irq as u32,
                is_wake_capable,
                is_shared,
                polarity,
                trigger,

                is_consumer: false, // assumed to be producer
            }))
        }
        _ => Err(AmlError::InvalidResourceDescriptor),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DMASupportedSpeed {
    CompatibilityMode,
    TypeA, // as described by the EISA
    TypeB,
    TypeF,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DMATransferTypePreference {
    _8BitOnly,
    _8And16Bit,
    _16Bit,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DMADescriptor {
    pub channel_mask: u8,
    pub supported_speeds: DMASupportedSpeed,
    pub is_bus_master: bool,
    pub transfer_type_preference: DMATransferTypePreference,
}

pub fn dma_format_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * DMA Descriptor Definition
     * Offset  Field Name
     * Byte 0  Value = 0x2A (00101010B) – Type = 0, Small item name = 0x5, Length = 2
     * Byte 1  DMA channel mask bits [7:0] (channels 0 – 7), _DMA
     *         Bit [0] is channel 0, etc.
     * Byte 2  Bit [7]           Reserved (must be 0)
     *         Bits [6:5]        DMA channel speed supported, _TYP
     *           00    Indicates compatibility mode
     *           01    Indicates Type A DMA as described in the EISA
     *           10    Indicates Type B DMA
     *           11    Indicates Type F
     *         Bits [4:3]        Ignored
     *         Bit [2]           Logical device bus master status, _BM
     *           0      Logical device is not a bus master
     *           1      Logical device is a bus master
     *         Bits [1:0]       DMA transfer type preference, _SIZ
     *           00    8-bit only
     *           01    8- and 16-bit
     *           10    16-bit only
     *           11    Reserved
     */
    if bytes.len() != 3 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let channel_mask = bytes[1];
    let options = bytes[2];
    let supported_speeds = match options.get_bits(5..=6) {
        0 => DMASupportedSpeed::CompatibilityMode,
        1 => DMASupportedSpeed::TypeA,
        2 => DMASupportedSpeed::TypeB,
        _ => DMASupportedSpeed::TypeF,
    };
    let is_bus_master = options.get_bit(2);
    let transfer_type_preference = match options.get_bits(0..=1) {
        0 => DMATransferTypePreference::_8BitOnly,
        1 => DMATransferTypePreference::_8And16Bit,
        2 => DMATransferTypePreference::_16Bit,
        // Hardening: the reserved value was `unimplemented!()`.
        _ => return Err(AmlError::InvalidResourceDescriptor),
    };

    Ok(Resource::Dma(DMADescriptor { channel_mask, supported_speeds, is_bus_master, transfer_type_preference }))
}

#[derive(Debug, PartialEq, Eq)]
pub struct IOPortDescriptor {
    pub decodes_full_address: bool,
    pub memory_range: (u16, u16),
    pub base_alignment: u8,
    pub range_length: u8,
}

fn io_port_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * I/O Port Descriptor Definition
     * Offset   Field Name                                  Definition
     * Byte 0   I/O Port Descriptor                         Value = 0x47 (01000111B) –
     *                                                      Type = 0, Small item name = 0x8, Length = 7
     * Byte 1   Information                                 Bits [7:1]     Reserved and must be 0
     *                                                      Bit [0]          (_DEC)
     *                                                        1    The logical device decodes 16-bit addresses
     *                                                        0    The logical device only decodes address bits[9:0]
     * Byte 2   Range minimum base address, _MIN bits[7:0]  Address bits [7:0] of the minimum base I/O address that the card may be configured for.
     * Byte 3   Range minimum base address, _MIN bits[15:8] Address bits [15:8] of the minimum base I/O address that the card may be configured for.
     * Byte 4   Range maximum base address, _MAX bits[7:0]  Address bits [7:0] of the maximum base I/O address that the card may be configured for.
     * Byte 5   Range maximum base address, _MAX bits[15:8] Address bits [15:8] of the maximum base I/O address that the card may be configured for.
     * Byte 6   Base alignment, _ALN                        Alignment for minimum base address, increment in 1-byte blocks.
     * Byte 7   Range length, _LEN                          The number of contiguous I/O ports requested.
     */
    if bytes.len() != 8 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let information = bytes[1];
    let decodes_full_address = information.get_bit(0);

    let memory_range_min = LittleEndian::read_u16(&bytes[2..=3]);
    let memory_range_max = LittleEndian::read_u16(&bytes[4..=5]);
    let memory_range = (memory_range_min, memory_range_max);

    let base_alignment = bytes[6];
    let range_length = bytes[7];

    Ok(Resource::IOPort(IOPortDescriptor { decodes_full_address, memory_range, base_alignment, range_length }))
}

fn extended_interrupt_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * --- Extended Interrupt Descriptor ---
     * Byte 3 contains the Interrupt Vector Flags:
     *      Bit 0: 1 if device consumes the resource, 0 if it produces it
     *      Bit 1: 1 if edge-triggered, 0 if level-triggered
     *      Bit 2: 1 = active-high, 0 = active-low
     *      Bit 3: 1 if interrupt is shared with other devices
     *      Bit 4: 1 if this interrupt is capable of waking the system, 0 if it is not
     * Byte 4 contains the number of interrupt numbers that follow. When this descriptor is
     * returned from `_CRS` or send to `_SRS`, this field must be 1.
     *
     * From Byte 5 onwards, there are `n` interrupt numbers, each of which is encoded as a
     * 4-byte little-endian number.
     *
     * NOTE: We only support the case where there is a single interrupt number.
     */
    if bytes.len() < 9 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    // Hardening: was `assert_eq!(.., 1)`: a list (as `_PRS` gives) is read as its first interrupt; none is an error.
    let number_of_interrupts = bytes[4] as usize;
    if number_of_interrupts == 0 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let irq = LittleEndian::read_u32(&[bytes[5], bytes[6], bytes[7], bytes[8]]);

    Ok(Resource::Irq(IrqDescriptor {
        is_consumer: bytes[3].get_bit(0),
        trigger: if bytes[3].get_bit(1) { InterruptTrigger::Edge } else { InterruptTrigger::Level },
        polarity: if bytes[3].get_bit(2) { InterruptPolarity::ActiveLow } else { InterruptPolarity::ActiveHigh },
        is_shared: bytes[3].get_bit(3),
        is_wake_capable: bytes[3].get_bit(4),
        irq,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parses_keyboard_crs() {
        let bytes: Vec<u8> = [
            // Generated from `iasl -l pc-bios_acpi-dsdt.asl`
            //
            //         315:                   IO (Decode16,
            //         316:                       0x0060,             // Range Minimum
            //         317:                       0x0060,             // Range Maximum
            //         318:                       0x01,               // Alignment
            //         319:                       0x01,               // Length
            //         320:                       )

            //    0000040A:  47 01 60 00 60 00 01 01     "G.`.`..."
            0x47, 0x01, 0x60, 0x00, 0x60, 0x00, 0x01, 0x01,
            //         321:                   IO (Decode16,
            //         322:                       0x0064,             // Range Minimum
            //         323:                       0x0064,             // Range Maximum
            //         324:                       0x01,               // Alignment
            //         325:                       0x01,               // Length
            //         326:                       )

            //    00000412:  47 01 64 00 64 00 01 01     "G.d.d..."
            0x47, 0x01, 0x64, 0x00, 0x64, 0x00, 0x01, 0x01,
            //         327:                   IRQNoFlags ()
            //         328:                       {1}

            //    0000041A:  22 02 00 ...............    "".."
            0x22, 0x02, 0x00, //    0000041D:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            Vec::from([
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x60, 0x60),
                    base_alignment: 1,
                    range_length: 1
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x64, 0x64),
                    base_alignment: 1,
                    range_length: 1
                }),
                Resource::Irq(IrqDescriptor {
                    is_consumer: false,
                    trigger: InterruptTrigger::Edge,
                    polarity: InterruptPolarity::ActiveHigh,
                    is_shared: false,
                    is_wake_capable: false,
                    irq: (1 << 1)
                })
            ])
        );
    }

    #[test]
    fn test_pci_crs() {
        let bytes: Vec<u8> = [
            // Generated from `iasl -l pc-bios_acpi-dsdt.asl`
            //
            //      98:               WordBusNumber (ResourceProducer, MinFixed, MaxFixed, PosDecode,
            //      99:                   0x0000,             // Granularity
            //     100:                   0x0000,             // Range Minimum
            //     101:                   0x00FF,             // Range Maximum
            //     102:                   0x0000,             // Translation Offset
            //     103:                   0x0100,             // Length
            //     104:                   ,, )

            // 000000F3:  88 0D 00 02 0C 00 00 00     "........"
            // 000000FB:  00 00 FF 00 00 00 00 01     "........"
            0x88, 0x0D, 0x00, 0x02, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x01,
            //     105:               IO (Decode16,
            //     106:                   0x0CF8,             // Range Minimum
            //     107:                   0x0CF8,             // Range Maximum
            //     108:                   0x01,               // Alignment
            //     109:                   0x08,               // Length
            //     110:                   )

            // 00000103:  47 01 F8 0C F8 0C 01 08     "G......."
            0x47, 0x01, 0xF8, 0x0C, 0xF8, 0x0C, 0x01, 0x08,
            //     111:               WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
            //     112:                   0x0000,             // Granularity
            //     113:                   0x0000,             // Range Minimum
            //     114:                   0x0CF7,             // Range Maximum
            //     115:                   0x0000,             // Translation Offset
            //     116:                   0x0CF8,             // Length
            //     117:                   ,, , TypeStatic, DenseTranslation)

            // 0000010B:  88 0D 00 01 0C 03 00 00     "........"
            // 00000113:  00 00 F7 0C 00 00 F8 0C     "........"
            0x88, 0x0D, 0x00, 0x01, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x00, 0xF7, 0x0C, 0x00, 0x00, 0xF8, 0x0C,
            //     118:               WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
            //     119:                   0x0000,             // Granularity
            //     120:                   0x0D00,             // Range Minimum
            //     121:                   0xFFFF,             // Range Maximum
            //     122:                   0x0000,             // Translation Offset
            //     123:                   0xF300,             // Length
            //     124:                   ,, , TypeStatic, DenseTranslation)

            // 0000011B:  88 0D 00 01 0C 03 00 00     "........"
            // 00000123:  00 0D FF FF 00 00 00 F3     "........"
            0x88, 0x0D, 0x00, 0x01, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x0D, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xF3,
            //     125:               DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, Cacheable, ReadWrite,
            //     126:                   0x00000000,         // Granularity
            //     127:                   0x000A0000,         // Range Minimum
            //     128:                   0x000BFFFF,         // Range Maximum
            //     129:                   0x00000000,         // Translation Offset
            //     130:                   0x00020000,         // Length
            //     131:                   ,, , AddressRangeMemory, TypeStatic)

            // 0000012B:  87 17 00 00 0C 03 00 00     "........"
            // 00000133:  00 00 00 00 0A 00 FF FF     "........"
            // 0000013B:  0B 00 00 00 00 00 00 00     "........"
            // 00000143:  02 00 ..................    ".."
            0x87, 0x17, 0x00, 0x00, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0xFF, 0xFF, 0x0B,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
            //     132:               DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, NonCacheable, ReadWrite,
            //     133:                   0x00000000,         // Granularity
            //     134:                   0xE0000000,         // Range Minimum
            //     135:                   0xFEBFFFFF,         // Range Maximum
            //     136:                   0x00000000,         // Translation Offset
            //     137:                   0x1EC00000,         // Length
            //     138:                   ,, _Y00, AddressRangeMemory, TypeStatic)

            // 00000145:  87 17 00 00 0C 01 00 00     "........"
            // 0000014D:  00 00 00 00 00 E0 FF FF     "........"
            // 00000155:  BF FE 00 00 00 00 00 00     "........"
            // 0000015D:  C0 1E ..................    ".."
            0x87, 0x17, 0x00, 0x00, 0x0C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0xFF, 0xFF, 0xBF,
            0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0x1E,
            // 0000015F:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            Vec::from([
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::BusNumberRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    granularity: 0,
                    address_range: (0x00, 0xFF),
                    translation_offset: 0,
                    length: 0x100
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0xCF8, 0xCF8),
                    base_alignment: 1,
                    range_length: 8
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::IORange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    granularity: 0,
                    address_range: (0x0000, 0x0CF7),
                    translation_offset: 0,
                    length: 0xCF8
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::IORange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    granularity: 0,
                    address_range: (0x0D00, 0xFFFF),
                    translation_offset: 0,
                    length: 0xF300
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::MemoryRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    granularity: 0,
                    address_range: (0xA0000, 0xBFFFF),
                    translation_offset: 0,
                    length: 0x20000
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::MemoryRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    granularity: 0,
                    address_range: (0xE0000000, 0xFEBFFFFF),
                    translation_offset: 0,
                    length: 0x1EC00000
                }),
            ])
        );
    }

    #[test]
    fn test_fdc_crs() {
        let bytes: Vec<u8> = [
            //         365:                   IO (Decode16,
            //         366:                       0x03F2,             // Range Minimum
            //         367:                       0x03F2,             // Range Maximum
            //         368:                       0x00,               // Alignment
            //         369:                       0x04,               // Length
            //         370:                       )

            //    0000047C:  47 01 F2 03 F2 03 00 04     "G......."
            0x47, 0x01, 0xF2, 0x03, 0xF2, 0x03, 0x00, 0x04,
            //         371:                   IO (Decode16,
            //         372:                       0x03F7,             // Range Minimum
            //         373:                       0x03F7,             // Range Maximum
            //         374:                       0x00,               // Alignment
            //         375:                       0x01,               // Length
            //         376:                       )

            //    00000484:  47 01 F7 03 F7 03 00 01     "G......."
            0x47, 0x01, 0xF7, 0x03, 0xF7, 0x03, 0x00, 0x01,
            //         377:                   IRQNoFlags ()
            //         378:                       {6}

            //    0000048C:  22 40 00 ...............    ""@."
            0x22, 0x40, 0x00,
            //         379:                   DMA (Compatibility, NotBusMaster, Transfer8, )
            //         380:                       {2}

            //    0000048F:  2A 04 00 ...............    "*.."
            0x2A, 0x04, 0x00, //    00000492:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            Vec::from([
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x03F2, 0x03F2),
                    base_alignment: 0,
                    range_length: 4
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x03F7, 0x03F7),
                    base_alignment: 0,
                    range_length: 1
                }),
                Resource::Irq(IrqDescriptor {
                    is_consumer: false,
                    trigger: InterruptTrigger::Edge,
                    polarity: InterruptPolarity::ActiveHigh,
                    is_shared: false,
                    is_wake_capable: false,
                    irq: (1 << 6)
                }),
                Resource::Dma(DMADescriptor {
                    channel_mask: 1 << 2,
                    supported_speeds: DMASupportedSpeed::CompatibilityMode,
                    is_bus_master: false,
                    transfer_type_preference: DMATransferTypePreference::_8BitOnly
                })
            ])
        );
    }
}
