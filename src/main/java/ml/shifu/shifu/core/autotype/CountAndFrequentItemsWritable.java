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
package ml.shifu.shifu.core.autotype;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.io.Writable;

/**
 * A mixed writable class to wrapper HyperLogLogPlus byte instance and frequent items together.
 * 
 * <p>
 * {@link #frequetItems} is used to check 0-1 variables which is not set to be categorical variables. THe size of it is
 * limited to {@link #FREQUET_ITEM_MAX_SIZE}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class CountAndFrequentItemsWritable implements Writable {

    static final int FREQUET_ITEM_MAX_SIZE = 20;
    /**
     * Serializing form for HyperLogLogPlus instance.
     */
    private byte[] hyperBytes;

    /**
     * Frequent items for one column, this set is limited to 10 and which is used so far to check 0-1 variables, such
     * 0-1 variables cannot be set to categorical variable.
     */
    private Set<String> frequetItems;

    public CountAndFrequentItemsWritable() {
    }

    public CountAndFrequentItemsWritable(byte[] hyperBytes, Set<String> frequetItems) {
        this.hyperBytes = hyperBytes;
        this.frequetItems = frequetItems;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.hadoop.io.Writable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        if(hyperBytes == null) {
            out.writeInt(0);
        } else {
            out.writeInt(hyperBytes.length);
            for(int i = 0; i < hyperBytes.length; i++) {
                out.writeByte(hyperBytes[i]);
            }
        }
        if(frequetItems == null) {
            out.writeInt(0);
        } else {
            int setSize = Math.min(frequetItems.size(), FREQUET_ITEM_MAX_SIZE);
            out.writeInt(setSize);
            Iterator<String> iter = frequetItems.iterator();
            int i = 0;
            while(i < setSize) {
                String unit = iter.next();
                if(unit == null) {
                    out.writeBoolean(false);
                } else {
                    out.writeBoolean(true);
                    out.writeUTF(unit);
                }
                i++;
            }
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
        hyperBytes = new byte[len];
        if(len != 0) {
            for(int i = 0; i < len; i++) {
                hyperBytes[i] = in.readByte();
            }
        }

        len = in.readInt();
        frequetItems = new HashSet<String>(len, 1f);
        if(len != 0) {
            for(int i = 0; i < len; i++) {
                if(in.readBoolean()) {
                    frequetItems.add(in.readUTF());
                }
            }
        }
    }

    /**
     * @return the hyperBytes
     */
    public byte[] getHyperBytes() {
        return hyperBytes;
    }

    /**
     * @param hyperBytes
     *            the hyperBytes to set
     */
    public void setHyperBytes(byte[] hyperBytes) {
        this.hyperBytes = hyperBytes;
    }

    /**
     * @return the frequetItems
     */
    public Set<String> getFrequetItems() {
        return frequetItems;
    }

    /**
     * @param frequetItems
     *            the frequetItems to set
     */
    public void setFrequetItems(Set<String> frequetItems) {
        this.frequetItems = frequetItems;
    }

}
