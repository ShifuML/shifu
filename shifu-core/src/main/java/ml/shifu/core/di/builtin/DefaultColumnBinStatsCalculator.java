/**
 * Copyright [2012-2014] eBay Software Foundation
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

package ml.shifu.core.di.builtin;


import ml.shifu.core.container.ColumnBinStatsResult;
import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.di.spi.ColumnBinStatsCalculator;

public class DefaultColumnBinStatsCalculator implements ColumnBinStatsCalculator {

    public ColumnBinStatsResult calculate(ColumnBinningResult binning) {
        ColumnBinStatsResult stats = new ColumnBinStatsResult();
        stats.setBinWoe(WOECalculator.calculate(binning.getBinCountPos().toArray(), binning.getBinCountNeg().toArray()));

        KSIVCalculator calculator = new KSIVCalculator();
        calculator.calculateKSIV(binning.getBinCountNeg(), binning.getBinCountPos());
        stats.setIv(calculator.getIV());
        stats.setKs(calculator.getKS());

        return stats;
    }

}
