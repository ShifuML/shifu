package ml.shifu.shifu.core.pmml.builder;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import ml.shifu.shifu.core.pmml.builder.impl.*;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;

import java.util.List;

/**
 * Created by zhanhu on 3/29/16.
 */
public class PMMLConstructorFactory {

    public static PMMLTranslator produce(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {

        AbstractPmmlElementCreator<Model> modelCreator = null;
        AbstractSpecifCreator specifCreator = null;
        if (ModelTrainConf.ALGORITHM.NN.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            modelCreator = new NNPmmlModelCreator(modelConfig, columnConfigList, isConcise);
            specifCreator = new NNSpecifCreator();
        } else if (ModelTrainConf.ALGORITHM.LR.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            modelCreator = new RegressionPmmlModelCreator(modelConfig, columnConfigList, isConcise);
            specifCreator = new RegressionSpecifCreator();
        } else {
            throw new RuntimeException("Model not supported: " + modelConfig.getTrain().getAlgorithm());
        }

        AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator =
                new DataDictionaryCreator(modelConfig, columnConfigList, isConcise);


        AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator =
                new MiningSchemaCreator(modelConfig, columnConfigList, isConcise);

        AbstractPmmlElementCreator<ModelStats> modelStatsCreator =
                new ModelStatsCreator(modelConfig, columnConfigList, isConcise);

        AbstractPmmlElementCreator<LocalTransformations> localTransformationsCreator = null;
        ModelNormalizeConf.NormType normType = modelConfig.getNormalizeType();
        if ( normType.equals(ModelNormalizeConf.NormType.WOE)
                || normType.equals(ModelNormalizeConf.NormType.WEIGHT_WOE) ) {
            localTransformationsCreator = new WoeLocalTransformCreator(modelConfig, columnConfigList, isConcise);
        } else {
            localTransformationsCreator = new ZscoreLocalTransformCreator(modelConfig, columnConfigList, isConcise);
        }

        return new PMMLTranslator(modelCreator,
                dataDictionaryCreator,
                miningSchemaCreator,
                modelStatsCreator,
                localTransformationsCreator,
                specifCreator);
    }
}
