package ml.shifu.core.di.spi;


import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningSchema;

import java.util.List;
import java.util.Map;

public interface TransformationExecutor {

    public Object transform(DerivedField derivedField, Object origin);

    public List<Object> transform(MiningSchema miningSchema, Map<FieldName, DerivedField> fieldNameToDerivedFieldMap, Map<FieldName, Integer> fieldNameToFieldNumberMap, List<Object> raw);
}
