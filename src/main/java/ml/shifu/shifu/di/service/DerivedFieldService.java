package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.DerivedFieldCreator;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.ModelStats;

public class DerivedFieldService {

    private DerivedFieldCreator derivedFieldCreator;

    @Inject
    public DerivedFieldService(DerivedFieldCreator derivedFieldCreator) {

        this.derivedFieldCreator = derivedFieldCreator;
    }

    public DerivedField exec(DataField dataField, ModelStats modelStats) {

        return derivedFieldCreator.create(dataField, modelStats);


    }

}
