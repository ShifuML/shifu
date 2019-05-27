package ml.shifu.shifu.core.varselect;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ColumnScore implements Writable, Cloneable {
    private int columnTag;
    private double weight;
    private double sensitivityScore;

    public int getColumnTag() {
        return columnTag;
    }

    public void setColumnTag(int columnTag) {
        this.columnTag = columnTag;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getSensitivityScore() {
        return sensitivityScore;
    }

    public void setSensitivityScore(double sensitivityScore) {
        this.sensitivityScore = sensitivityScore;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnTag);
        out.writeDouble(this.weight);
        out.writeDouble(this.sensitivityScore);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnTag = in.readInt();
        this.weight = in.readDouble();
        this.sensitivityScore = in.readDouble();
    }

    @Override
    public String toString() {
        return columnTag + "," + weight + "," + sensitivityScore;
    }

    @Override
    public ColumnScore clone() {
        ColumnScore copy = new ColumnScore();
        copy.setColumnTag(this.columnTag);
        copy.setWeight(this.weight);
        copy.setSensitivityScore(this.sensitivityScore);
        return copy;
    }
}
