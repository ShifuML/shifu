package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLDerivedFieldCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.ModelStats;

public class PMMLDerivedFieldService {

    private PMMLDerivedFieldCreator PMMLDerivedFieldCreator;

    @Inject
    public PMMLDerivedFieldService(PMMLDerivedFieldCreator PMMLDerivedFieldCreator) {

        this.PMMLDerivedFieldCreator = PMMLDerivedFieldCreator;
    }

    public DerivedField exec(DataField dataField, ModelStats modelStats, Params params) {

        return PMMLDerivedFieldCreator.create(dataField, modelStats, params);


    }

}
