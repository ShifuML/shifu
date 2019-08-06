package ml.shifu.shifu.core.shuffle;

import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

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

public class UpdateWeightDataMapper extends AbstractDataMapper {

    protected double rblRatio;
    protected int targetIndex;
    protected String delimiter;

    public UpdateWeightDataMapper(double rblRatio, int targetIndex, String delimiter) {
        this.rblRatio = rblRatio;
        this.targetIndex = targetIndex;
        this.delimiter = delimiter;
    }

    @Override
    public void mapData(Mapper.Context context, Text line, int shuffleSize)
            throws IOException, InterruptedException {
        String[] fields = CommonUtils.split(line.toString(), this.delimiter);
        if(POS_TAG.equals(CommonUtils.trimTag(fields[targetIndex]))) { // positive record, update weight
            // update the weight
            fields[fields.length - 1] = updateWeight(fields[fields.length - 1], rblRatio);
            // combine fields to new output
            String outputLine = StringUtils.join(fields, delimiter);

            IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(shuffleSize));
            context.write(shuffleIndex, new Text(outputLine));
        } else { // negative record, keep it is
            IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(shuffleSize));
            context.write(shuffleIndex, line);
        }
    }

    private String updateWeight(String field, double rblRatio) {
        double weight;
        try {
            weight = Double.parseDouble(field);
        } catch (Exception e) {
            // it won't be here!
            weight = 1.0d;
        }
        return Double.toString(weight * rblRatio);
    }
}
