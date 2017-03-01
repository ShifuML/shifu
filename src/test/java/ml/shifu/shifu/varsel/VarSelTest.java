/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.varsel;

import java.io.IOException;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.VariableSelector.Tuple;
import ml.shifu.shifu.util.CommonUtils;

import org.testng.annotations.Test;

public class VarSelTest {

    @Test
    public void testParento() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                getClass().getResource("/camdttest/config/ModelConfig.json").toString(), SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                getClass().getResource("/camdttest/config/ColumnConfig.json").toString(), SourceType.LOCAL);
        VariableSelector vs = new VariableSelector(modelConfig, columnConfigList);
        // for(ColumnConfig columnConfig: columnConfigList) {
        // Double ks = columnConfig.getKs();
        // Double iv = columnConfig.getIv();
        // System.out.println(columnConfig.getColumnNum() + "\t" + (ks == null ? 0d : ks) + "\t"
        // + (iv == null ? 0d : iv));
        // }
        List<Tuple> sortByParetoCC = vs.sortByParetoCC(columnConfigList);
        for(Tuple tuple: sortByParetoCC) {
            System.out.println(tuple.columnNum);
        }
    }

}
