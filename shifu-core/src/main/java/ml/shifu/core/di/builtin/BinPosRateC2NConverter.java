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

import ml.shifu.core.container.CategoricalValueObject;
import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.container.NumericalValueObject;

public class BinPosRateC2NConverter {

    private ColumnBinningResult columnBinningResult;

    public void setColumnBinningResult(ColumnBinningResult columnBinningResult) {
        this.columnBinningResult = columnBinningResult;
    }

    public NumericalValueObject convert(CategoricalValueObject cvo) {
        if (columnBinningResult == null) {
            throw new RuntimeException("No ColumnBinningResult specified. Call setColumnBinningResult() first.");
        }
        int index = columnBinningResult.getBinCategory().indexOf(cvo.getValue());

        NumericalValueObject nvo = new NumericalValueObject();

        if (index == -1) {
            // TODO: how to deal with missing data
            nvo.setValue(0.0);
        } else {
            nvo.setValue(columnBinningResult.getBinPosRate().get(index));
        }
        nvo.setWeight(cvo.getWeight());


        return nvo;
    }
}
