/*
 * Copyright [2012-2015] PayPal Software Foundation
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
import java.util.Iterator;

import ml.shifu.shifu.util.SizeEstimator;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.buffer.BufferedMLDataSet;

/**
 * A hybrid data set combining {@link BasicMLDataSet} and {@link BufferedMLDataSet} together.
 * 
 * <p>
 * With this data set, element is added firstly in memory, if over {@link #maxByteSize} then element will be added into
 * disk.
 * 
 * <p>
 * This data set provide a very important feature to make in memory computing more stable. Even for some cases no enough
 * memory, memory and disk will be leveraged together to accelerate computing.
 * 
 * <p>
 * Example almost same as {@link BufferedMLDataSet}:
 * 
 * <pre>
 * MemoryDiskMLDataSet dataSet = new MemoryDiskMLDataSet(400, "a.txt");
 * dataSet.beginLoad(10, 1);
 * dataSet.add(pair);
 * dataSet.endLoad();
 * 
 * Iterator iterator = dataSet.iterator();
 * while(iterator.hasNext()) {
 *     MLDataPair next = iterator.next();
 *     ...
 * }
 * 
 * dataSet.close();
 * </pre>
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MemoryDiskMLDataSet implements MLDataSet {

    /**
     * Max bytes located in memory.
     */
    private long maxByteSize = Long.MAX_VALUE;

    /**
     * Current bytes for added elements.
     */
    private long byteSize = 0;

    /**
     * Memory data set which type is {@link BasicMLDataSet}
     */
    private MLDataSet memoryDataSet;

    /**
     * Disk data set which type is {@link BufferedMLDataSet}
     */
    private MLDataSet diskDataSet;

    /**
     * Input variable count
     */
    private int inputCount;

    /**
     * Output target count.
     */
    private int outputCount;

    /**
     * File name which is used for {@link #diskDataSet}
     */
    private String fileName;

    /**
     * How many records located into memory
     */
    private long memoryCount = 0L;

    /**
     * How many records located into disk
     */
    private long diskCount = 0L;

    /**
     * Constructor with {@link #fileName}, {@link #inputCount} and {@link #outputCount}
     * 
     * @param fileName
     *            the fileName
     * @param inputCount
     *            the inputCount
     * @param outputCount
     *            the inputCount
     */
    public MemoryDiskMLDataSet(String fileName, int inputCount, int outputCount) {
        this.memoryDataSet = new BasicMLDataSet();
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.fileName = fileName;
    }

    /**
     * Constructor with {@link #maxByteSize} and {@link #fileName}
     * 
     * @param maxByteSize
     *            the maxByteSize
     * @param fileName
     *            the fileName
     */
    public MemoryDiskMLDataSet(long maxByteSize, String fileName) {
        this.maxByteSize = maxByteSize;
        this.memoryDataSet = new BasicMLDataSet();
        this.fileName = fileName;
    }

    /**
     * Constructor with {@link #maxByteSize}, {@link #fileName}, {@link #inputCount} and {@link #outputCount}.
     * 
     * @param maxByteSize
     *            the maxByteSize
     * @param fileName
     *            the fileName
     * @param inputCount
     *            the inputCount
     * @param outputCount
     *            the inputCount
     */
    public MemoryDiskMLDataSet(long maxByteSize, String fileName, int inputCount, int outputCount) {
        this.maxByteSize = maxByteSize;
        this.memoryDataSet = new BasicMLDataSet();
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.fileName = fileName;
    }

    /**
     * Setting input variable size and output target size.
     * 
     * @param inputSize
     *            input variable size
     * @param idealSize
     *            output target size
     */
    public final void beginLoad(final int inputSize, final int idealSize) {
        this.inputCount = inputSize;
        this.outputCount = idealSize;
        if(this.diskDataSet != null) {
            ((BufferedMLDataSet) this.diskDataSet).beginLoad(this.inputCount, this.outputCount);
        }
    }

    /**
     * This method should be called once all the data has been loaded. The underlying file will be closed. The binary
     * fill will then be opened for reading.
     */
    public final void endLoad() {
        if(this.diskDataSet != null) {
            ((BufferedMLDataSet) this.diskDataSet).endLoad();
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Iterable#iterator()
     */
    @Override
    public Iterator<MLDataPair> iterator() {
        return new Iterator<MLDataPair>() {

            private Iterator<MLDataPair> iter1 = MemoryDiskMLDataSet.this.memoryDataSet.iterator();

            private Iterator<MLDataPair> iter2 = MemoryDiskMLDataSet.this.diskDataSet == null ? null
                    : MemoryDiskMLDataSet.this.diskDataSet.iterator();

            /**
             * If iterating in memory data set
             */
            private boolean isMemoryHasNext = false;

            /**
             * If iterating in disk data set
             */
            private boolean isDiskHasNext = false;

            @Override
            public boolean hasNext() {
                boolean hasNext = iter1.hasNext();
                if(hasNext) {
                    isMemoryHasNext = true;
                    isDiskHasNext = false;
                    return hasNext;
                }
                hasNext = iter2 == null ? false : iter2.hasNext();
                if(hasNext) {
                    isMemoryHasNext = false;
                    isDiskHasNext = true;
                } else {
                    isMemoryHasNext = false;
                    isDiskHasNext = false;
                }
                return hasNext;
            }

            @Override
            public MLDataPair next() {
                if(isMemoryHasNext) {
                    return iter1.next();
                }
                if(isDiskHasNext) {
                    if(iter2 != null) {
                        return iter2.next();
                    }
                }
                return null;
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }
        };
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#getIdealSize()
     */
    @Override
    public int getIdealSize() {
        return this.outputCount;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#getInputSize()
     */
    @Override
    public int getInputSize() {
        return this.inputCount;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#isSupervised()
     */
    @Override
    public boolean isSupervised() {
        return this.memoryDataSet.isSupervised();
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#getRecordCount()
     */
    @Override
    public long getRecordCount() {
        long count = this.memoryDataSet.getRecordCount();
        if(this.diskDataSet != null) {
            count += this.diskDataSet.getRecordCount();
        }
        return count;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#getRecord(long, org.encog.ml.data.MLDataPair)
     */
    @Override
    public void getRecord(long index, MLDataPair pair) {
        if(index < this.memoryCount) {
            this.memoryDataSet.getRecord(index, pair);
        } else {
            this.diskDataSet.getRecord(index - this.memoryCount, pair);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#openAdditional()
     */
    @Override
    public MLDataSet openAdditional() {
        throw new UnsupportedOperationException();
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#add(org.encog.ml.data.MLData)
     */
    @Override
    public void add(MLData data) {
        long currentSize = SizeEstimator.estimate(data);
        if(this.byteSize + currentSize < this.maxByteSize) {
            this.byteSize += currentSize;
            this.memoryCount += 1l;
            this.memoryDataSet.add(data);
        } else {
            if(this.diskDataSet == null) {
                this.diskDataSet = new BufferedMLDataSet(new File(this.fileName));
                ((BufferedMLDataSet) this.diskDataSet).beginLoad(this.inputCount, this.outputCount);
            }
            this.byteSize += currentSize;
            this.diskCount += 1l;
            this.diskDataSet.add(data);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#add(org.encog.ml.data.MLData, org.encog.ml.data.MLData)
     */
    @Override
    public void add(MLData inputData, MLData idealData) {
        long currentSize = SizeEstimator.estimate(inputData) + SizeEstimator.estimate(idealData);
        if(this.byteSize + currentSize < this.maxByteSize) {
            this.byteSize += currentSize;
            this.memoryCount += 1l;
            this.memoryDataSet.add(inputData, idealData);
        } else {
            if(this.diskDataSet == null) {
                this.diskDataSet = new BufferedMLDataSet(new File(this.fileName));
                ((BufferedMLDataSet) this.diskDataSet).beginLoad(this.inputCount, this.outputCount);
            }
            this.byteSize += currentSize;
            this.diskCount += 1l;
            this.diskDataSet.add(inputData, idealData);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#add(org.encog.ml.data.MLDataPair)
     */
    @Override
    public void add(MLDataPair inputData) {
        long currentSize = SizeEstimator.estimate(inputData);
        if(this.byteSize + currentSize < this.maxByteSize) {
            this.byteSize += currentSize;
            this.memoryCount += 1l;
            this.memoryDataSet.add(inputData);
        } else {
            if(this.diskDataSet == null) {
                this.diskDataSet = new BufferedMLDataSet(new File(this.fileName));
                ((BufferedMLDataSet) this.diskDataSet).beginLoad(this.inputCount, this.outputCount);
            }
            this.byteSize += currentSize;
            this.diskCount += 1l;
            this.diskDataSet.add(inputData);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.data.MLDataSet#close()
     */
    @Override
    public void close() {
        this.memoryDataSet.close();
        if(this.diskDataSet != null) {
            this.diskDataSet.close();
        }
    }

    /**
     * @return the memoryCount
     */
    public long getMemoryCount() {
        return memoryCount;
    }

    /**
     * @return the diskCount
     */
    public long getDiskCount() {
        return diskCount;
    }

}
