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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.encog.ml.data.buffer.BufferedMLDataSet;

/**
 * Copy from {@link BufferedMLDataSet} to support float type data.
 */
public class BufferedFloatMLDataSet implements FloatMLDataSet, Serializable {

    /**
     * The version.
     */
    private static final long serialVersionUID = 2577778772598513566L;

    /**
     * Error message for ADD.
     */
    public static final String ERROR_ADD = "Add can only be used after calling beginLoad.";

    /**
     * Error message for REMOVE.
     */
    public static final String ERROR_REMOVE = "Remove is not supported for BufferedNeuralDataSet.";

    /**
     * True, if we are in the process of loading.
     */
    private transient boolean loading;

    /**
     * The file being used.
     */
    private File file;

    /**
     * The EGB file we are working wtih.
     */
    private transient EncogFloatEGBFile egb;

    /**
     * Additional sets that were opened.
     */
    private transient List<BufferedFloatMLDataSet> additional = new ArrayList<BufferedFloatMLDataSet>();

    /**
     * The owner.
     */
    private transient BufferedFloatMLDataSet owner;

    /**
     * Construct the dataset using the specified binary file.
     * 
     * @param binaryFile
     *            The file to use.
     */
    public BufferedFloatMLDataSet(final File binaryFile) {
        this.file = binaryFile;
        this.egb = new EncogFloatEGBFile(binaryFile);
        if(file.exists()) {
            this.egb.open();
        }
    }

    /**
     * Open the binary file for reading.
     */
    public final void open() {
        this.egb.open();
    }

    /**
     * @return An iterator.
     */
    @Override
    public final Iterator<FloatMLDataPair> iterator() {
        return new BufferedFloatDataSetIterator(this);
    }

    /**
     * @return The record count.
     */
    @Override
    public final long getRecordCount() {
        if(this.egb == null) {
            return 0;
        } else {
            return this.egb.getNumberOfRecords();
        }
    }

    /**
     * Read an individual record.
     * 
     * @param index
     *            The zero-based index. Specify 0 for the first record, 1 for
     *            the second, and so on.
     * @param pair
     *            THe data to read.
     */
    @Override
    public final void getRecord(final long index, final FloatMLDataPair pair) {
        this.egb.setLocation((int) index);
        float[] inputTarget = pair.getInputArray();
        this.egb.read(inputTarget);

        if(pair.getIdealArray() != null) {
            float[] idealTarget = pair.getIdealArray();
            this.egb.read(idealTarget);
        }

        this.egb.read();
    }

    /**
     * @return An additional training set.
     */
    @Override
    public final BufferedFloatMLDataSet openAdditional() {
        BufferedFloatMLDataSet result = new BufferedFloatMLDataSet(this.file);
        result.setOwner(this);
        this.additional.add(result);
        return result;
    }

    /**
     * Add only input data, for an unsupervised dataset.
     * 
     * @param data1
     *            The data to be added.
     */
    public final void add(final FloatMLData data1) {
        if(!this.loading) {
            throw new RuntimeException(BufferedFloatMLDataSet.ERROR_ADD);
        }

        egb.write(data1.getData());
        egb.write(1.0f);
    }

    /**
     * Add both the input and ideal data.
     * 
     * @param inputData
     *            The input data.
     * @param idealData
     *            The ideal data.
     */
    public final void add(final FloatMLData inputData, final FloatMLData idealData) {

        if(!this.loading) {
            throw new RuntimeException(BufferedFloatMLDataSet.ERROR_ADD);
        }

        this.egb.write(inputData.getData());
        this.egb.write(idealData.getData());
        this.egb.write((float) 1.0f);
    }

    /**
     * Add a data pair of both input and ideal data.
     * 
     * @param pair
     *            The pair to add.
     */
    public final void add(final FloatMLDataPair pair) {
        if(!this.loading) {
            throw new RuntimeException(BufferedFloatMLDataSet.ERROR_ADD);
        }

        this.egb.write(pair.getInputArray());
        this.egb.write(pair.getIdealArray());
        this.egb.write(pair.getSignificance());

    }

    /**
     * Close the dataset.
     */
    @Override
    public final void close() {

        Object[] obj = this.additional.toArray();

        for(int i = 0; i < obj.length; i++) {
            BufferedFloatMLDataSet set = (BufferedFloatMLDataSet) obj[i];
            set.close();
        }

        this.additional.clear();

        if(this.owner != null) {
            this.owner.removeAdditional(this);
        }

        this.egb.close();
        this.egb = null;
    }

    /**
     * @return The ideal data size.
     */
    @Override
    public final int getIdealSize() {
        if(this.egb == null) {
            return 0;
        } else {
            return this.egb.getIdealCount();
        }
    }

    /**
     * @return The input data size.
     */
    @Override
    public final int getInputSize() {
        if(this.egb == null) {
            return 0;
        } else {
            return this.egb.getInputCount();
        }
    }

    /**
     * @return True if this dataset is supervised.
     */
    @Override
    public final boolean isSupervised() {
        if(this.egb == null) {
            return false;
        } else {
            return this.egb.getIdealCount() > 0;
        }
    }

    /**
     * @return If this dataset was created by openAdditional, the set that
     *         created this object is the owner. Return the owner.
     */
    public final BufferedFloatMLDataSet getOwner() {
        return owner;
    }

    /**
     * Set the owner of this dataset.
     * 
     * @param theOwner
     *            The owner.
     */
    public final void setOwner(final BufferedFloatMLDataSet theOwner) {
        this.owner = theOwner;
    }

    /**
     * Remove an additional dataset that was created.
     * 
     * @param child
     *            The additional dataset to remove.
     */
    public final void removeAdditional(final BufferedFloatMLDataSet child) {
        synchronized(this) {
            this.additional.remove(child);
        }
    }

    /**
     * Begin loading to the binary file. After calling this method the add
     * methods may be called.
     * 
     * @param inputSize
     *            The input size.
     * @param idealSize
     *            The ideal size.
     */
    public final void beginLoad(final int inputSize, final int idealSize) {
        this.egb.create(inputSize, idealSize);
        this.loading = true;
    }

    /**
     * This method should be called once all the data has been loaded. The
     * underlying file will be closed. The binary fill will then be opened for
     * reading.
     */
    public final void endLoad() {
        if(!this.loading) {
            throw new RuntimeException("Must call beginLoad, before endLoad.");
        }

        this.egb.close();

        open();
    }

    /**
     * @return The binary file used.
     */
    public final File getFile() {
        return this.file;
    }

    /**
     * @return The EGB file to use.
     */
    public final EncogFloatEGBFile getEGB() {
        return this.egb;
    }

    /**
     * Load the binary dataset to memory. Memory access is faster.
     * 
     * @return A memory dataset.
     */
    public final FloatMLDataSet loadToMemory() {
        BasicFloatMLDataSet result = new BasicFloatMLDataSet();

        for(FloatMLDataPair pair: this) {
            result.add(pair);
        }

        return result;
    }

    /**
     * Load the specified training set.
     * 
     * @param training
     *            The training set to load.
     */
    public final void load(final FloatMLDataSet training) {
        beginLoad(training.getInputSize(), training.getIdealSize());
        for(final FloatMLDataPair pair: training) {
            add(pair);
        }
        endLoad();
    }

}
