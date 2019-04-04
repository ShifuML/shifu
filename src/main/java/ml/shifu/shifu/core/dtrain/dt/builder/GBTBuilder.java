package ml.shifu.shifu.core.dtrain.dt.builder;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

import java.util.List;
import java.util.Random;

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

@SuppressWarnings("unused")
public class GBTBuilder extends TreeBuilder {

    /**
     * Learning rate GBDT.
     */
    private double learningRate = 0.1d;

    /**
     * By default in GBDT, sample with replacement is enabled, but looks sometimes good performance with replacement &
     * GBDT
     */
    private boolean gbdtSampleWithReplacement = false;

    /**
     * Drop out rate for gbdt to drop trees in training. http://xgboost.readthedocs.io/en/latest/tutorials/dart.html
     */
    private double dropOutRate = 0.0;

    /**
     * Random object to drop out trees, work with {@link #dropOutRate}
     */
    private Random dropOutRandom = new Random(System.currentTimeMillis() + 5000L);

    /**
     * A flag means current worker is fail over task and gbdt predict value needs to be recovered. After data recovered,
     * such falg should reset to false
     */
    private boolean isNeedRecoverGBDTPredict = false;

    public GBTBuilder(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

}
