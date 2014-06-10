package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.builtin.TransformationExecutor;
import ml.shifu.shifu.di.spi.DerivedFieldCreator;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.ModelStats;

public class TransformationInitService {

    private DerivedFieldCreator derivedFieldCreator;

    @Inject
    public TransformationInitService(DerivedFieldCreator derivedFieldCreator) {

        this.derivedFieldCreator = derivedFieldCreator;
    }

    public DerivedField exec(DataField dataField, ModelStats modelStats) {

        return derivedFieldCreator.create(dataField, modelStats);
    }

}
