"""
Simple icon generator using only built-in libraries
Creates a basic ICO file for Windows
"""
import struct

def create_simple_ico():
    """Create a simple ICO file using binary format"""
    # ICO file format: Header + Directory + Image data
    
    # For simplicity, we'll create a 32x32 and 16x16 icon
    # Using a simple colored square with "S" pattern
    
    # 32x32 icon data (RGBA, 4 bytes per pixel)
    size32 = 32
    icon32_data = bytearray()
    
    # Create a gradient background (indigo/purple)
    for y in range(size32):
        for x in range(size32):
            # Calculate distance from center
            cx, cy = size32 // 2, size32 // 2
            dx = x - cx
            dy = y - cy
            dist = (dx*dx + dy*dy) ** 0.5
            max_dist = (cx*cx + cy*cy) ** 0.5
            
            # Gradient from center (lighter) to edges (darker)
            factor = 1 - (dist / max_dist) * 0.5
            r = int(99 * factor)
            g = int(102 * factor)
            b = int(241 * factor)
            
            # Draw "S" pattern (simple checkered pattern in center)
            if (8 <= x <= 24 and 8 <= y <= 24):
                # Draw white "S" shape
                if ((x < 16 and y < 16) or (x >= 16 and y >= 16)):
                    r, g, b = 255, 255, 255
            
            icon32_data.extend([b, g, r, 255])  # BGRA format
    
    # Compress/encode as BMP format for ICO
    # BMP header for 32x32 RGBA
    bmp_header_size = 54
    image_data_size = len(icon32_data)
    file_size = bmp_header_size + image_data_size
    
    bmp32 = bytearray()
    # BMP file header
    bmp32.extend(b'BM')  # Signature
    bmp32.extend(struct.pack('<I', file_size))  # File size
    bmp32.extend(struct.pack('<I', 0))  # Reserved
    bmp32.extend(struct.pack('<I', bmp_header_size))  # Offset to image data
    # DIB header (BITMAPINFOHEADER)
    bmp32.extend(struct.pack('<I', 40))  # Header size
    bmp32.extend(struct.pack('<i', size32))  # Width
    bmp32.extend(struct.pack('<i', size32 * 2))  # Height (double for ICO)
    bmp32.extend(struct.pack('<H', 1))  # Planes
    bmp32.extend(struct.pack('<H', 32))  # Bits per pixel
    bmp32.extend(struct.pack('<I', 0))  # Compression
    bmp32.extend(struct.pack('<I', image_data_size))  # Image size
    bmp32.extend(struct.pack('<I', 0))  # X pixels per meter
    bmp32.extend(struct.pack('<I', 0))  # Y pixels per meter
    bmp32.extend(struct.pack('<I', 0))  # Colors used
    bmp32.extend(struct.pack('<I', 0))  # Important colors
    # Image data (flip vertically for BMP)
    for y in range(size32 - 1, -1, -1):
        bmp32.extend(icon32_data[y * size32 * 4:(y + 1) * size32 * 4])
    
    # 16x16 icon (simpler version)
    size16 = 16
    icon16_data = bytearray()
    for y in range(size16):
        for x in range(size16):
            factor = 0.7
            r = int(99 * factor)
            g = int(102 * factor)
            b = int(241 * factor)
            if (4 <= x <= 12 and 4 <= y <= 12):
                if ((x < 8 and y < 8) or (x >= 8 and y >= 8)):
                    r, g, b = 255, 255, 255
            icon16_data.extend([b, g, r, 255])
    
    bmp16 = bytearray()
    file_size16 = bmp_header_size + len(icon16_data)
    bmp16.extend(b'BM')
    bmp16.extend(struct.pack('<I', file_size16))
    bmp16.extend(struct.pack('<I', 0))
    bmp16.extend(struct.pack('<I', bmp_header_size))
    bmp16.extend(struct.pack('<I', 40))
    bmp16.extend(struct.pack('<i', size16))
    bmp16.extend(struct.pack('<i', size16 * 2))
    bmp16.extend(struct.pack('<H', 1))
    bmp16.extend(struct.pack('<H', 32))
    bmp16.extend(struct.pack('<I', 0))
    bmp16.extend(struct.pack('<I', len(icon16_data)))
    bmp16.extend(struct.pack('<I', 0))
    bmp16.extend(struct.pack('<I', 0))
    bmp16.extend(struct.pack('<I', 0))
    bmp16.extend(struct.pack('<I', 0))
    for y in range(size16 - 1, -1, -1):
        bmp16.extend(icon16_data[y * size16 * 4:(y + 1) * size16 * 4])
    
    # ICO file structure
    ico_data = bytearray()
    # ICO header
    ico_data.extend(struct.pack('<H', 0))  # Reserved (must be 0)
    ico_data.extend(struct.pack('<H', 1))  # Type (1 = ICO)
    ico_data.extend(struct.pack('<H', 2))  # Number of images
    
    # Directory entry 1 (16x16)
    ico_data.extend(b'\x10')  # Width (16)
    ico_data.extend(b'\x10')  # Height (16)
    ico_data.extend(b'\x00')  # Color palette (0 = no palette)
    ico_data.extend(b'\x00')  # Reserved
    ico_data.extend(struct.pack('<H', 1))  # Color planes
    ico_data.extend(struct.pack('<H', 32))  # Bits per pixel
    ico_data.extend(struct.pack('<I', len(bmp16)))  # Size of image data
    ico_data.extend(struct.pack('<I', 22))  # Offset to image data (6 + 16*2)
    
    # Directory entry 2 (32x32)
    ico_data.extend(b'\x20')  # Width (32)
    ico_data.extend(b'\x20')  # Height (32)
    ico_data.extend(b'\x00')
    ico_data.extend(b'\x00')
    ico_data.extend(struct.pack('<H', 1))
    ico_data.extend(struct.pack('<H', 32))
    ico_data.extend(struct.pack('<I', len(bmp32)))
    ico_data.extend(struct.pack('<I', 22 + len(bmp16)))
    
    # Image data
    ico_data.extend(bmp16)
    ico_data.extend(bmp32)
    
    # Write ICO file
    with open('stocker_icon.ico', 'wb') as f:
        f.write(ico_data)
    
    print("Simple icon created: stocker_icon.ico")

if __name__ == "__main__":
    try:
        create_simple_ico()
        print("Icon generation complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

