/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml.builder;

import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.TreeEnsemblePMMLTranslator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import ml.shifu.shifu.core.pmml.builder.impl.*;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;

/**
 * Created by zhanhu on 3/29/16.
 */
public class PMMLConstructorFactory {

    public static PMMLTranslator produce(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            boolean isConcise, boolean isOutBaggingToOne) {

        AbstractPmmlElementCreator<Model> modelCreator = null;
        AbstractSpecifCreator specifCreator = null;
        if(ModelTrainConf.ALGORITHM.NN.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            modelCreator = new NNPmmlModelCreator(modelConfig, columnConfigList, isConcise);
            specifCreator = new NNSpecifCreator();
        } else if(ModelTrainConf.ALGORITHM.LR.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            modelCreator = new RegressionPmmlModelCreator(modelConfig, columnConfigList, isConcise);
            specifCreator = new RegressionSpecifCreator();
        } else if(ModelTrainConf.ALGORITHM.GBT.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())
                || ModelTrainConf.ALGORITHM.RF.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            TreeEnsemblePmmlCreator gbtmodelCreator = new TreeEnsemblePmmlCreator(modelConfig, columnConfigList);
            AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator = new DataDictionaryCreator(modelConfig,
                    columnConfigList);
            AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator = new TreeModelMiningSchemaCreator(
                    modelConfig, columnConfigList);
            return new TreeEnsemblePMMLTranslator(gbtmodelCreator, dataDictionaryCreator, miningSchemaCreator);
        } else {
            throw new RuntimeException("Model not supported: " + modelConfig.getTrain().getAlgorithm());
        }

        AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator = new DataDictionaryCreator(modelConfig,
                columnConfigList, isConcise);

        AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator = new MiningSchemaCreator(modelConfig,
                columnConfigList, isConcise);

        AbstractPmmlElementCreator<ModelStats> modelStatsCreator = new ModelStatsCreator(modelConfig, columnConfigList,
                isConcise);

        AbstractPmmlElementCreator<LocalTransformations> localTransformationsCreator = null;
        ModelNormalizeConf.NormType normType = modelConfig.getNormalizeType();
        if(normType.equals(ModelNormalizeConf.NormType.WOE) || normType.equals(ModelNormalizeConf.NormType.WEIGHT_WOE)) {
            localTransformationsCreator = new WoeLocalTransformCreator(modelConfig, columnConfigList, isConcise);
        } else if(normType == ModelNormalizeConf.NormType.WOE_ZSCORE
                || normType == ModelNormalizeConf.NormType.WOE_ZSCALE) {
            localTransformationsCreator = new WoeZscoreLocalTransformCreator(modelConfig, columnConfigList, isConcise,
                    false);
        } else if(normType == ModelNormalizeConf.NormType.WEIGHT_WOE_ZSCORE
                || normType == ModelNormalizeConf.NormType.WEIGHT_WOE_ZSCALE) {
            localTransformationsCreator = new WoeZscoreLocalTransformCreator(modelConfig, columnConfigList, isConcise,
                    true);
        } else {
            localTransformationsCreator = new ZscoreLocalTransformCreator(modelConfig, columnConfigList, isConcise);
        }

        return new PMMLTranslator(modelCreator, dataDictionaryCreator, miningSchemaCreator, modelStatsCreator,
                localTransformationsCreator, specifCreator, isOutBaggingToOne);
    }
}
