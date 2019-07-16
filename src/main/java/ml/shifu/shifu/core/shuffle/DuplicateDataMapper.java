package ml.shifu.shifu.core.shuffle;

import ml.shifu.shifu.util.CommonUtils;
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

public class DuplicateDataMapper extends UpdateWeightDataMapper {

    private boolean duplicatePos;

    public DuplicateDataMapper(double rblRatio, int targetIndex, String delimiter) {
        super(rblRatio, targetIndex, delimiter);

        if(this.rblRatio > 1.0d) {
            duplicatePos = true;
        } else {
            duplicatePos = false;
            this.rblRatio = 1.0d / this.rblRatio;
        }
    }

    @Override
    public void mapData(Mapper.Context context, Text line, int shuffleSize)
            throws IOException, InterruptedException {
        String[] fields = CommonUtils.split(line.toString(), this.delimiter);
        if(this.duplicatePos == POS_TAG.equals(CommonUtils.trimTag(fields[targetIndex]))) {
            double totalRatio = rblRatio;
            while(totalRatio > 0) {
                double seed = rd.nextDouble();
                if(seed < totalRatio) {
                    IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(shuffleSize));
                    context.write(shuffleIndex, line);
                }
                totalRatio -= 1.0d;
            }
        } else {
            IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(shuffleSize));
            context.write(shuffleIndex, line);
        }
    }
}
