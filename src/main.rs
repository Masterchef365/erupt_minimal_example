use anyhow::{format_err, Result};
use erupt::{
    cstr,
    utils::{
        allocator::{Allocator, AllocatorCreateInfo, MemoryTypeFinder},
    },
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::ffi::CString;

fn main() -> Result<()> {
    let entry = EntryLoader::new()?;

    // Instance
    let name = CString::new("TensorSludge")?;
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&name)
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(&name)
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Instance and device layers and extensions
    let mut instance_layers = Vec::new();
    let mut instance_extensions = Vec::new();
    let mut device_layers = Vec::new();
    let device_extensions = Vec::new();

    // Vulkan layers and extensions
    if cfg!(debug_assertions) {
        const LAYER_KHRONOS_VALIDATION: *const i8 = cstr!("VK_LAYER_KHRONOS_validation");
        instance_extensions
            .push(erupt::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
        device_layers.push(LAYER_KHRONOS_VALIDATION);
    }

    // Instance creation
    let create_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let instance = InstanceLoader::new(&entry, &create_info, None)?;

    // Hardware selection
    let (queue_family_index, physical_device) = select_device(&instance)?;

    // Create logical device and queues
    let create_info = [vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0])];

    let physical_device_features = vk::PhysicalDeviceFeaturesBuilder::new();
    let create_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&create_info)
        .enabled_features(&physical_device_features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device = DeviceLoader::new(&instance, physical_device, &create_info, None)?;
    //let queue = unsafe { device.get_device_queue(queue_family_index, 0, None) };

    // Allocator
    let mut allocator =
        Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).result()?;

    // Just allocates a buffer of the given size
    let mut alloc = move |buffer_size: u64| -> Result<_> {
        let create_info = vk::BufferCreateInfoBuilder::new()
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(buffer_size);

        let buffer = unsafe { device.create_buffer(&create_info, None, None) }.result()?;
        let data = allocator
            .allocate(&device, buffer, MemoryTypeFinder::gpu_only())
            .result()?;
        Ok(data)
    };

    const IMG_WIDTH: usize = 28;
    const IMG_SIZE: usize = IMG_WIDTH * IMG_WIDTH;
    const HIDDEN_L1: usize = 128;
    const BATCHES: usize = 50;

    let buffer_size = BATCHES * IMG_SIZE * HIDDEN_L1 * std::mem::size_of::<f32>();

    if std::env::args().count() > 1 {
        let a = alloc(buffer_size as _)?;
        let b = alloc(buffer_size as _)?;
        drop((a, b));
    } else {
        alloc((buffer_size * 2) as _)?;
    }

    Ok(())
}

fn select_device(instance: &InstanceLoader) -> Result<(u32, vk::PhysicalDevice)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
    for device in physical_devices {
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(device, None) };
        for (family, properites) in families.iter().enumerate() {
            if properites.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((family as u32, device));
            }
        }
    }
    Err(format_err!("No suitable device found"))
}
