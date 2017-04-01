/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dataset;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Copy from {@link EncogFloatEGBFile} to support float type data.
 */
public class EncogFloatEGBFile {

    /**
     * The size of a float.
     */
    public static final int FLOAT_SIZE = Float.SIZE / 8;

    /**
     * The size of the file header.
     */
    public static final int HEADER_SIZE = EncogFloatEGBFile.FLOAT_SIZE * 2 * 3;

    /**
     * The file that we are working with.
     */
    private final File file;

    /**
     * The number of input values per record.
     */
    private int inputCount;

    /**
     * The number of ideal values per record.
     */
    private int idealCount;

    /**
     * The underlying file.
     */
    private RandomAccessFile raf;

    /**
     * The file channel used.
     */
    private FileChannel fc;

    /**
     * A byte buffer to hold the header.
     */
    private final ByteBuffer headerBuffer;

    /**
     * A byte buffer to hold the records.
     */
    private ByteBuffer recordBuffer;

    /**
     * The number of values in a record, this is the input and ideal combined.
     */
    private int recordCount;

    /**
     * The size of a record.
     */
    private int recordSize;

    /**
     * The number of records int he file.
     */
    private int numberOfRecords;

    /**
     * Construct an EGB file.
     * 
     * @param theFile
     *            The file.
     */
    public EncogFloatEGBFile(final File theFile) {
        this.file = theFile;
        this.headerBuffer = ByteBuffer.allocate(EncogFloatEGBFile.HEADER_SIZE);
    }

    /**
     * Add a column.
     * 
     * @param col
     *            THe column to add.
     * @param isInput
     *            Is this an input column?
     */
    public final void addColumn(final int col, final boolean isInput) {
        try {
            // process the file
            // allocate buffers
            final ByteBuffer readBuffer = ByteBuffer.allocate(EncogFloatEGBFile.FLOAT_SIZE * 1024);
            final ByteBuffer writeBuffer = ByteBuffer.allocate(EncogFloatEGBFile.FLOAT_SIZE * 1024);

            readBuffer.clear();
            writeBuffer.clear();
            readBuffer.order(ByteOrder.LITTLE_ENDIAN);
            writeBuffer.order(ByteOrder.LITTLE_ENDIAN);

            long readLocation = EncogFloatEGBFile.HEADER_SIZE;
            long writeLocation = EncogFloatEGBFile.HEADER_SIZE;
            int recordOffset = 0;

            this.fc.position(readLocation);
            this.fc.read(readBuffer);
            readLocation = this.fc.position();
            readBuffer.rewind();

            boolean done = false;
            int count = 0;

            do {
                // if there is more to read, then process it
                if(readBuffer.hasRemaining()) {
                    final float d = readBuffer.getFloat();

                    // If this is the column to insert, add a zero
                    if(recordOffset == col) {
                        // do we need to cycle the write buffer?
                        writeLocation = checkWrite(writeBuffer, writeLocation);
                        writeBuffer.putFloat(0);
                    }

                    // write the existing value
                    writeLocation = checkWrite(writeBuffer, writeLocation);
                    writeBuffer.putFloat(d);

                    // keep track of where we are in a record.
                    recordOffset++;
                    if(recordOffset >= this.recordCount) {
                        recordOffset = 0;
                        count++;
                        // are we done?
                        if(count >= this.numberOfRecords) {
                            done = true;
                        }
                    }
                } else {
                    // read more
                    readBuffer.clear();
                    readBuffer.order(ByteOrder.LITTLE_ENDIAN);

                    this.fc.position(readLocation);
                    this.fc.read(readBuffer);
                    readLocation = this.fc.position();
                    readBuffer.rewind();
                }
            } while(!done);

            // write any remaining data in the write buffer
            if(writeBuffer.position() > 0) {
                writeBuffer.flip();
                this.fc.write(writeBuffer, writeLocation);
            }

            // does it fall inside of input or ideal?
            if(isInput) {
                this.inputCount++;
                this.recordCount++;
            } else {
                this.idealCount++;
                this.recordCount++;
            }

            this.recordCount = this.inputCount + this.idealCount + 1;
            this.recordSize = this.recordCount * EncogFloatEGBFile.FLOAT_SIZE;

            // adjust file size
            this.raf.setLength((this.numberOfRecords * this.recordSize) + EncogFloatEGBFile.HEADER_SIZE);

        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Add a row.
     * 
     * @param row
     *            Where to add the row.
     */
    public final void addRow(final int row) {
        try {
            this.numberOfRecords++;

            this.raf.setLength((this.numberOfRecords * this.recordSize) + EncogFloatEGBFile.HEADER_SIZE);

            for(int i = this.numberOfRecords - 1; i >= row; i--) {
                final int s = EncogFloatEGBFile.HEADER_SIZE + (this.recordSize * i);
                final int t = EncogFloatEGBFile.HEADER_SIZE + (this.recordSize * i) + this.recordSize;

                clear();
                this.fc.read(this.recordBuffer, s);
                this.recordBuffer.flip();
                this.fc.write(this.recordBuffer, t);
            }

            clear();
            for(int i = 0; i < this.recordCount - 1; i++) {
                this.recordBuffer.putFloat(0f);
            }
            this.recordBuffer.putFloat(1.0f);
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer, EncogFloatEGBFile.HEADER_SIZE + (this.recordSize * row));

        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }

    }

    /**
     * Calculate the index for the specified row.
     * 
     * @param row
     *            The row to calculate for.
     * @return The index.
     */
    private long calculateIndex(final long row) {
        return (long) EncogFloatEGBFile.HEADER_SIZE + (row * (long) this.recordSize);
    }

    /**
     * Read a row and column.
     * 
     * @param row
     *            The row, or record, to read.
     * @param col
     *            The column to read.
     * @return THe value read.
     */
    private int calculateIndex(final int row, final int col) {
        return EncogFloatEGBFile.HEADER_SIZE + (row * this.recordSize) + (col * EncogFloatEGBFile.FLOAT_SIZE);
    }

    /**
     * Check a write, make sure there is enough room to write.
     * 
     * @param writeBuffer
     *            The buffer.
     * @param inWriteLocation
     *            The write location.
     * @return The new write location.
     * @throws IOException
     *             If an IO error occurs.
     */
    private long checkWrite(final ByteBuffer writeBuffer, final long inWriteLocation) throws IOException {
        long writeLocation = inWriteLocation;

        if(!writeBuffer.hasRemaining()) {
            this.fc.position(writeLocation);
            writeBuffer.flip();
            this.fc.write(writeBuffer);
            writeLocation = this.fc.position();
            writeBuffer.clear();
            writeBuffer.order(ByteOrder.LITTLE_ENDIAN);
        }

        return writeLocation;
    }

    /**
     * Clear the record buffer.
     */
    private void clear() {
        this.recordBuffer.clear();
        this.recordBuffer.order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Close the file.
     */
    public final void close() {
        try {
            if(this.raf != null) {
                this.raf.close();
                this.raf = null;
            }
            if(this.fc != null) {
                this.fc.close();
                this.fc = null;
            }
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Create a new RGB file.
     * 
     * @param theInputCount
     *            The input count.
     * @param theIdealCount
     *            The ideal count.
     */
    public final void create(final int theInputCount, final int theIdealCount) {
        try {
            this.inputCount = theInputCount;
            this.idealCount = theIdealCount;

            final float[] input = new float[inputCount];
            final float[] ideal = new float[idealCount];

            this.raf = new RandomAccessFile(this.file, "rw");
            this.fc = this.raf.getChannel();

            this.headerBuffer.clear();
            this.headerBuffer.order(ByteOrder.LITTLE_ENDIAN);

            this.headerBuffer.put((byte) 'E');
            this.headerBuffer.put((byte) 'N');
            this.headerBuffer.put((byte) 'C');
            this.headerBuffer.put((byte) 'O');
            this.headerBuffer.put((byte) 'G');
            this.headerBuffer.put((byte) '-');
            this.headerBuffer.put((byte) '0');
            this.headerBuffer.put((byte) '0');

            this.headerBuffer.putFloat(input.length);
            this.headerBuffer.putFloat(ideal.length);

            this.numberOfRecords = 0;
            this.recordCount = this.inputCount + this.idealCount + 1;
            this.recordSize = this.recordCount * EncogFloatEGBFile.FLOAT_SIZE;
            this.recordBuffer = ByteBuffer.allocate(this.recordSize);

            this.headerBuffer.flip();
            this.fc.write(this.headerBuffer);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Delete a column.
     * 
     * @param col
     *            The column to delete.
     */
    public final void deleteCol(final int col) {
        try {
            // process the file

            // allocate buffers
            final ByteBuffer readBuffer = ByteBuffer.allocate(EncogFloatEGBFile.FLOAT_SIZE * 1024);
            final ByteBuffer writeBuffer = ByteBuffer.allocate(EncogFloatEGBFile.FLOAT_SIZE * 1024);

            readBuffer.clear();
            writeBuffer.clear();
            readBuffer.order(ByteOrder.LITTLE_ENDIAN);
            writeBuffer.order(ByteOrder.LITTLE_ENDIAN);

            long readLocation = EncogFloatEGBFile.HEADER_SIZE;
            long writeLocation = EncogFloatEGBFile.HEADER_SIZE;
            int recordOffset = 0;

            this.fc.position(readLocation);
            this.fc.read(readBuffer);
            readLocation = this.fc.position();
            readBuffer.rewind();

            boolean done = false;
            int count = 0;

            do {
                // if there is more to read, then process it
                if(readBuffer.hasRemaining()) {
                    final float d = readBuffer.getFloat();
                    // skip the specified column, as we write
                    if(recordOffset != col) {
                        writeLocation = checkWrite(writeBuffer, writeLocation);
                        writeBuffer.putFloat(d);
                    }

                    // keep track of where we are in a record.
                    recordOffset++;
                    if(recordOffset >= this.recordCount) {
                        recordOffset = 0;
                        count++;
                        // are we done?
                        if(count >= this.numberOfRecords) {
                            done = true;
                        }
                    }
                } else {
                    // read more
                    readBuffer.clear();
                    readBuffer.order(ByteOrder.LITTLE_ENDIAN);

                    this.fc.position(readLocation);
                    this.fc.read(readBuffer);
                    readLocation = this.fc.position();
                    readBuffer.rewind();
                }
            } while(!done);

            // write any remaining data in the write buffer
            if(writeBuffer.position() > 0) {
                writeBuffer.flip();
                this.fc.write(writeBuffer, writeLocation);
            }

            // does it fall inside of input or ideal?
            if(col < this.inputCount) {
                this.inputCount--;
                this.recordCount--;
            } else {
                this.idealCount--;
                this.recordCount--;
            }

            this.recordCount = this.inputCount + this.idealCount + 1;
            this.recordSize = this.recordCount * EncogFloatEGBFile.FLOAT_SIZE;

            // adjust file size
            this.raf.setLength((this.numberOfRecords * this.recordSize) + EncogFloatEGBFile.HEADER_SIZE);

        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Delete a row.
     * 
     * @param row
     *            The row to delete.
     */
    public final void deleteRow(final int row) {
        try {
            for(int i = row; i < this.numberOfRecords - 1; i++) {
                final int s = EncogFloatEGBFile.HEADER_SIZE + (this.recordSize * i) + this.recordSize;
                final int t = EncogFloatEGBFile.HEADER_SIZE + (this.recordSize * i);

                clear();
                this.fc.read(this.recordBuffer, s);
                this.recordBuffer.flip();
                this.fc.write(this.recordBuffer, t);
            }

            this.numberOfRecords--;

            this.raf.setLength((this.numberOfRecords * this.recordSize) + EncogFloatEGBFile.HEADER_SIZE);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * @return the fc
     */
    public final FileChannel getFc() {
        return this.fc;
    }

    /**
     * @return the file
     */
    public final File getFile() {
        return this.file;
    }

    /**
     * @return the headerBuffer
     */
    public final ByteBuffer getHeaderBuffer() {
        return this.headerBuffer;
    }

    /**
     * @return the idealCount
     */
    public final int getIdealCount() {
        return this.idealCount;
    }

    /**
     * @return the inputCount
     */
    public final int getInputCount() {
        return this.inputCount;
    }

    /**
     * @return the numberOfRecords
     */
    public final int getNumberOfRecords() {
        return this.numberOfRecords;
    }

    /**
     * @return the raf
     */
    public final RandomAccessFile getRaf() {
        return this.raf;
    }

    /**
     * @return the recordBuffer
     */
    public final ByteBuffer getRecordBuffer() {
        return this.recordBuffer;
    }

    /**
     * @return the recordCount
     */
    public final int getRecordCount() {
        return this.recordCount;
    }

    /**
     * @return the recordSize
     */
    public final int getRecordSize() {
        return this.recordSize;
    }

    /**
     * OPen an existing EGB file.
     */
    public final void open() {
        try {
            this.raf = new RandomAccessFile(this.file, "rw");
            this.fc = this.raf.getChannel();

            this.headerBuffer.clear();
            this.headerBuffer.order(ByteOrder.LITTLE_ENDIAN);

            boolean isEncogFile = true;

            this.fc.read(this.headerBuffer);
            this.headerBuffer.position(0);

            isEncogFile = isEncogFile ? this.headerBuffer.get() == 'E' : false;
            isEncogFile = isEncogFile ? this.headerBuffer.get() == 'N' : false;
            isEncogFile = isEncogFile ? this.headerBuffer.get() == 'C' : false;
            isEncogFile = isEncogFile ? this.headerBuffer.get() == 'O' : false;
            isEncogFile = isEncogFile ? this.headerBuffer.get() == 'G' : false;
            isEncogFile = isEncogFile ? this.headerBuffer.get() == '-' : false;

            if(!isEncogFile) {
                throw new RuntimeException("File is not a valid Encog binary file:" + this.file.toString());
            }

            final char v1 = (char) this.headerBuffer.get();
            final char v2 = (char) this.headerBuffer.get();
            final String versionStr = "" + v1 + v2;

            try {
                final int version = Integer.parseInt(versionStr);
                if(version > 0) {
                    throw new RuntimeException("File is from a newer version of Encog than is currently in use.");
                }
            } catch (final NumberFormatException ex) {
                throw new RuntimeException("File has invalid version number.");
            }

            this.inputCount = (int) this.headerBuffer.getFloat();
            this.idealCount = (int) this.headerBuffer.getFloat();

            this.recordCount = this.inputCount + this.idealCount + 1;
            this.recordSize = this.recordCount * EncogFloatEGBFile.FLOAT_SIZE;
            if(this.recordSize == 0) {
                this.numberOfRecords = 0;
            } else {
                this.numberOfRecords = (int) ((this.file.length() - EncogFloatEGBFile.HEADER_SIZE) / this.recordSize);
            }

            this.recordBuffer = ByteBuffer.allocate(this.recordSize);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }

    }

    /**
     * Read a single float.
     * 
     * @return The float read.
     */
    public final float read() {
        try {
            clear();
            this.recordBuffer.limit(EncogFloatEGBFile.FLOAT_SIZE);
            this.fc.read(this.recordBuffer);
            this.recordBuffer.position(0);
            return this.recordBuffer.getFloat();
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Read an array of floats.
     * 
     * @param d
     *            The array to read into.
     */
    public final void read(final float[] d) {
        try {
            clear();
            this.recordBuffer.limit(EncogFloatEGBFile.FLOAT_SIZE * d.length);
            this.fc.read(this.recordBuffer);
            this.recordBuffer.position(0);
            for(int i = 0; i < d.length; i++) {
                d[i] = this.recordBuffer.getFloat();
            }
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Read a float array at the specified record.
     * 
     * @param row
     *            The record to read.
     * @param d
     *            The array to read into.
     */
    public final void read(final int row, final float[] d) {
        try {
            clear();
            this.recordBuffer.limit(EncogFloatEGBFile.FLOAT_SIZE * d.length);
            this.fc.read(this.recordBuffer, calculateIndex(row));
            this.recordBuffer.position(0);

            for(int i = 0; i < this.recordCount; i++) {
                d[i] = this.recordBuffer.getFloat();
            }
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Read a row and column.
     * 
     * @param row
     *            The row to read.
     * @param col
     *            The column to read.
     * @return The value read.
     */
    public final float read(final int row, final int col) {
        try {
            clear();
            this.recordBuffer.limit(EncogFloatEGBFile.FLOAT_SIZE);
            this.fc.read(this.recordBuffer, calculateIndex(row, col));
            this.recordBuffer.position(0);
            return this.recordBuffer.getFloat(0);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }

    }

    /**
     * Set the current location to the specified row.
     * 
     * @param row
     *            The row.
     */
    public final void setLocation(final int row) {
        try {
            this.fc.position(calculateIndex(row));
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Write a byte.
     * 
     * @param b
     *            The byte to write.
     */
    public final void write(final byte b) {
        try {
            clear();
            this.recordBuffer.put(b);
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Write an array.
     * 
     * @param v
     *            The array to write.
     */
    public final void write(final float[] v) {
        try {
            clear();
            for(final float element: v) {
                this.recordBuffer.putFloat(element);
            }
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Write an array at the specified record.
     * 
     * @param row
     *            The record to write.
     * @param v
     *            The array to write.
     */
    public final void write(final int row, final float[] v) {
        try {
            clear();
            for(final float element: v) {
                this.recordBuffer.putFloat(element);
            }
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }

    }

    /**
     * Write the specified row and column.
     * 
     * @param row
     *            The row.
     * @param col
     *            The column.
     * @param v
     *            The value.
     */
    public final void write(final int row, final int col, final float v) {
        try {
            clear();
            this.recordBuffer.putFloat(v);
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer, calculateIndex(row, col));
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public void write(float d) {
        try {
            clear();
            this.recordBuffer.putFloat(d);
            this.recordBuffer.flip();
            this.fc.write(this.recordBuffer);
        } catch (final IOException ex) {
            throw new RuntimeException(ex);
        }
    }

}
