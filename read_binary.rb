#!/usr/bin/env ruby

filename = File.expand_path("~/Documents/NMIST/train-images-idx3-ubyte")

begin
  File.open(filename, "rb") do |file|
    # Read and parse header (first 16 bytes)
    header = file.read(16).bytes
    magic_number = header[0..3].pack('C*').unpack('N')[0]
    num_images = header[4..7].pack('C*').unpack('N')[0]
    num_rows = header[8..11].pack('C*').unpack('N')[0]
    num_cols = header[12..15].pack('C*').unpack('N')[0]

    puts "File Header Information:"
    puts "Magic Number: #{magic_number} (should be 2051 for images)"
    puts "Number of Images: #{num_images}"
    puts "Number of Rows: #{num_rows}"
    puts "Number of Columns: #{num_cols}"
    puts "\nFirst 50 bytes of image data:"
    
    # Read first 50 bytes of actual image data
    image_data = file.read(50).bytes
    image_data.each_with_index do |byte, index|
      print "#{byte} "
      puts if (index + 1) % 10 == 0  # newline every 10 numbers for readability
    end
  end
rescue Errno::ENOENT
  puts "Error: File '#{filename}' not found"
rescue => e
  puts "Error reading file: #{e.message}"
end
