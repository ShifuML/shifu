package ml.shifu.core.di.builtin.derivedField;

import ml.shifu.core.di.spi.DerivedFieldCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

public class PassThroughDerivedFieldCreator implements DerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {

        String suffix = params.get("suffix", "_Transformed").toString();


        FieldRef fieldRef = new FieldRef();
        fieldRef.setField(dataField.getName());

        DerivedField derivedField = new DerivedField();
        derivedField.setName(new FieldName(dataField.getName().getValue() + suffix));
        derivedField.setExpression(fieldRef);

        return derivedField;
    }
}
