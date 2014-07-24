package ml.shifu.core.di.builtin.derivedfield;

import ml.shifu.core.di.spi.PMMLDerivedFieldCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

public class PassThroughPMMLDerivedFieldCreator implements PMMLDerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {

        String suffix;
        if (params != null && params.containsKey("suffix")) {
            suffix = params.get("suffix").toString();
        } else {
            suffix = "_transformed";
        }


        FieldRef fieldRef = new FieldRef();
        fieldRef.setField(dataField.getName());

        DerivedField derivedField = new DerivedField();
        derivedField.setName(new FieldName(dataField.getName().getValue() + suffix));
        derivedField.setExpression(fieldRef);

        return derivedField;
    }
}
