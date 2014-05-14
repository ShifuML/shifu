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

package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.ColumnCatBinningCalculator;
import ml.shifu.shifu.di.spi.ColumnNumBinningCalculator;
import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ColumnBinningService {

    private static Logger log = LoggerFactory.getLogger(ColumnBinningService.class);

    private ColumnNumBinningCalculator nBinning;
    private ColumnCatBinningCalculator cBinning;

    @Inject
    public ColumnBinningService(ColumnNumBinningCalculator nBinning, ColumnCatBinningCalculator cBinning) {

        log.debug("Dependency Injected: ColumnNumBinningCalculator => " + nBinning.getClass().toString());
        log.debug("Dependency Injected: ColumnCatBinningCalculator => " + cBinning.getClass().toString());
        this.nBinning = nBinning;
        this.cBinning = cBinning;
    }

    public ColumnBinningResult getNumericalResult(List<NumericalValueObject> nvoList, int maxNumBins) {
        return nBinning.calculate(nvoList, maxNumBins);
    }

    public ColumnBinningResult getCategoricalResult(List<CategoricalValueObject> cvoList) {

        return cBinning.calculate(cvoList);

    }

}
