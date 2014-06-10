package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.builtin.TransformationExecutor;
import org.dmg.pmml.DerivedField;

public class TransformationExecService {

    private TransformationExecutor transformationExecutor;

    @Inject
    public TransformationExecService(TransformationExecutor transformationExecutor) {

        this.transformationExecutor = transformationExecutor;
    }

    public Object exec(DerivedField derivedField, Object origin) {

        return transformationExecutor.transform(derivedField, origin);
    }

}
