package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.TransformationExecutor;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningSchema;

import java.util.List;
import java.util.Map;

public class TransformationExecService {

    private TransformationExecutor transformationExecutor;

    @Inject
    public TransformationExecService(TransformationExecutor transformationExecutor) {

        this.transformationExecutor = transformationExecutor;
    }

    public Object exec(DerivedField derivedField, Object origin) {

        return transformationExecutor.transform(derivedField, origin);
    }

    public List<Object> exec(MiningSchema miningSchema, Map<FieldName, DerivedField> fieldNameToDerivedFieldMap, Map<FieldName, Integer> fieldNameToFieldNumberMap, List<Object> raw) {

        return transformationExecutor.transform(miningSchema, fieldNameToDerivedFieldMap, fieldNameToFieldNumberMap, raw);
    }
}
