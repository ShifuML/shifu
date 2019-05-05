package ml.shifu.shifu.core.varselect;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

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

public class VarSelPerfGenerator {

    private static Logger LOG = LoggerFactory.getLogger(VarSelPerfGenerator.class);

    private List<ColumnScore> columnScores;
    private OpMetric opMetric;
    private double opUnit;

    private long totalUnitCount;
    private double totalWeightSum;

    private long totalPosUnitCount;
    private double totalPosWeightSum;

    public VarSelPerfGenerator(List<ColumnScore> columnScores, OpMetric opMetric, double opUnit) {
        this.columnScores = columnScores;
        this.opMetric = opMetric;
        this.opUnit = opUnit;

        this.totalUnitCount = columnScores.size();
        this.totalWeightSum = 0.0d;
        this.totalPosUnitCount = 0;
        this.totalPosWeightSum = 0.0d;
        for (ColumnScore columnScore : columnScores) {
            this.totalWeightSum += columnScore.getWeight();
            if ( columnScore.getColumnTag() > 0 ) {
                this.totalPosUnitCount ++;
                this.totalPosWeightSum += columnScore.getWeight();
            }
        }

        LOG.info("totalUnitCount = {}, while totalPosUnitCount = {}", this.totalUnitCount, this.totalPosUnitCount);
        LOG.info("totalWeightSum = {}, while totalPosWeightSum = {}", this.totalWeightSum, this.totalPosWeightSum);
    }

    public double getSensitivityPerf() {
        Collections.sort(columnScores, new Comparator<ColumnScore>() {
            @Override
            public int compare(ColumnScore from, ColumnScore to) {
                return Double.compare(to.getSensitivityScore(), from.getSensitivityScore());
            }
        });

        return genOpPerf();
    }

    private double genOpPerf() {
        double totalAll = (opMetric.equals(OpMetric.ACTION_RATE)
                ? this.totalUnitCount : this.totalWeightSum);
        double totalPositive = (opMetric.equals(OpMetric.ACTION_RATE)
                ? this.totalPosUnitCount : this.totalPosWeightSum);

        double opPerf = 0.0d;
        double opPoint = 0.0d;

        double accumulateTotal = 0.0d;
        double accumulatePositive = 0.0d;

        if ( opMetric.equals(OpMetric.ACTION_RATE) ) {
            for (ColumnScore columnScore: this.columnScores) {
                accumulateTotal += 1.0d;
                if(columnScore.getColumnTag() > 0) {
                    accumulatePositive += 1.0d;
                }

                opPoint = (accumulateTotal / totalAll);
                if (opPoint > this.opUnit) {
                    opPerf = (accumulatePositive / totalPositive);
                    break;
                }
            }
        } else {
            for (ColumnScore columnScore: this.columnScores) {
                accumulateTotal += columnScore.getWeight();
                if(columnScore.getColumnTag() > 0) {
                    accumulatePositive += columnScore.getWeight();
                }

                opPoint = (accumulateTotal / totalAll);
                if (opPoint > this.opUnit) {
                    opPerf = (accumulatePositive / totalPositive);
                    break;
                }

            }
        }

        return opPerf;
    }
}
