package ml.shifu.core.di.builtin.miningschema;

import ml.shifu.core.di.spi.PMMLMiningSchemaCreator;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

import java.util.HashSet;
import java.util.Set;

public class DefaultPMMLMiningSchemaCreator implements PMMLMiningSchemaCreator {

    public MiningSchema create(PMML pmml, Params params) throws Exception {

        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName").toString());

        MiningSchema miningSchema = new MiningSchema();
        model.setMiningSchema(miningSchema);

        Set<String> targetNameSet = new HashSet<String>();
        for (Target target : model.getTargets().getTargets()) {
            targetNameSet.add(target.getField().getValue());
        }


        for (DataField dataField : pmml.getDataDictionary().getDataFields()) {
            Params fieldParams = params.getFieldConfig(dataField.getName().getValue()).getParams();


            MiningField miningField = new MiningField();
            miningField.setName(dataField.getName());

            miningField.setOptype(dataField.getOptype());
            if (targetNameSet.contains(miningField.getName().getValue())) {
                miningField.setUsageType(FieldUsageType.TARGET);
            } else {
                miningField.setUsageType(FieldUsageType.valueOf(fieldParams.get("usageType").toString().toUpperCase()));
            }

            miningSchema.withMiningFields(miningField);
        }


        return miningSchema;
    }

}
