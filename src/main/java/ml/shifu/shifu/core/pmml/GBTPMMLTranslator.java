package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import ml.shifu.shifu.core.pmml.builder.impl.GBTPmmlCreator;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import ml.shifu.shifu.core.TreeModel;

import java.util.List;

public class GBTPMMLTranslator extends PMMLTranslator {
        
    private GBTPmmlCreator modelCreator;
    private AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator;
    private AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator;

    public GBTPMMLTranslator(GBTPmmlCreator modelCreator,
                          AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator,
                          AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator) {
        super();
        this.modelCreator = modelCreator;
        this.dataDictionaryCreator = dataDictionaryCreator;
        this.miningSchemaCreator = miningSchemaCreator;
    }
    public PMML build(BasicML basicML) {
            PMML pmml = new PMML();
            pmml.setDataDictionary(dataDictionaryCreator.build());
            List<Model> models = pmml.getModels();
            Model miningModel = modelCreator.convert(((TreeModel)basicML).getIndependentTreeModel());
            models.add(miningModel);
            return pmml;
    }
}
