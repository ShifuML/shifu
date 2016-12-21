package ml.shifu.shifu.core.pmml;

import java.util.List;

import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.impl.TreeEnsemblePmmlCreator;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;

public class TreeEnsemblePMMLTranslator extends PMMLTranslator {
        
    private TreeEnsemblePmmlCreator modelCreator;
    private AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator;
    @SuppressWarnings("unused")
    private AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator;

    public TreeEnsemblePMMLTranslator(TreeEnsemblePmmlCreator modelCreator,
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
