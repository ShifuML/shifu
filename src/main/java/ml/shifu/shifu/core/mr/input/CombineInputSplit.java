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
package ml.shifu.shifu.core.mr.input;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

/**
 * {@link InputSplit} implementation to combine multiple .
 * 
 * <p>
 * For worker, input {@link #fileSplits} are included, here <code>FileSplit</code> array is used to support combining
 * <code>FileSplit</code>s in one task.
 */
public class CombineInputSplit extends InputSplit implements Writable {

    /**
     * File splits used for the task. Using array here to make support combining small files into one GuaguaInputSplit.
     */
    private FileSplit[] fileSplits;

    /**
     * Default constructor without any setting.
     */
    public CombineInputSplit() {
    }

    /**
     * Constructor with {@link #fileSplits} settings.
     * 
     * @param fileSplits
     *            File splits used for mapper task.
     */
    public CombineInputSplit(FileSplit... fileSplits) {
        this.fileSplits = fileSplits;
    }

    /**
     * Constructor with one FileSplit settings.
     * 
     * @param fileSplit
     *            File split used for mapper task.
     */
    public CombineInputSplit(FileSplit fileSplit) {
        this(new FileSplit[] { fileSplit });
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.hadoop.io.Writable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        int length = this.getFileSplits().length;
        out.writeInt(length);
        for(int i = 0; i < length; i++) {
            this.getFileSplits()[i].write(out);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.hadoop.io.Writable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        int len = in.readInt();
        FileSplit[] splits = new FileSplit[len];
        for(int i = 0; i < len; i++) {
            splits[i] = new FileSplit(null, 0, 0, (String[]) null);
            splits[i].readFields(in);
        }
        this.setFileSplits(splits);
    }

    /**
     * For master split, use <code>Long.MAX_VALUE</code> as its length to make it is the first task for Hadoop job. It
     * is convenient for users to check master in Hadoop UI.
     */
    @Override
    public long getLength() throws IOException, InterruptedException {
        long len = 0;
        for(FileSplit split: this.getFileSplits()) {
            len += split.getLength();
        }
        return len;
    }

    /**
     * Data locality functions, return all hosts for all file splits.
     */
    @Override
    public String[] getLocations() throws IOException, InterruptedException {
        if(this.getFileSplits() == null || this.getFileSplits().length == 0) {
            return new String[0];
        }

        List<String> hosts = new ArrayList<String>();
        for(FileSplit fileSplit: this.getFileSplits()) {
            if(fileSplit != null) {
                hosts.addAll(Arrays.asList(fileSplit.getLocations()));
            }
        }

        return hosts.toArray(new String[0]);
    }

    public FileSplit[] getFileSplits() {
        return fileSplits;
    }

    public void setFileSplits(FileSplit[] fileSplits) {
        this.fileSplits = fileSplits;
    }

    @Override
    public String toString() {
        return String.format("CombineInputSplit [fileSplit=%s]", Arrays.toString(this.fileSplits));
    }

}
