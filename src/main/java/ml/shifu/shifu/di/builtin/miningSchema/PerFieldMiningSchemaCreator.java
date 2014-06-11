package ml.shifu.shifu.di.builtin.miningSchema;

import ml.shifu.shifu.di.spi.MiningSchemaCreator;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

import java.util.HashSet;
import java.util.Set;
import java.util.Map;

public class PerFieldMiningSchemaCreator implements MiningSchemaCreator {

    public MiningSchema create(Model model, PMML pmml, RequestObject req) {

        DataDictionary dataDictionary = pmml.getDataDictionary();

        MiningSchema miningSchema = new MiningSchema();

        Set<String> targetNameSet = new HashSet<String>();
        for (Target target : model.getTargets().getTargets()) {
            targetNameSet.add(target.getField().getValue());
        }

        for (DataField dataField : dataDictionary.getDataFields()) {

            Params params = req.getFieldParams(dataField.getName().getValue());

            MiningField miningField = new MiningField();
            miningField.setName(dataField.getName());

            miningField.setOptype(dataField.getOptype());
            if (targetNameSet.contains(miningField.getName().getValue())) {
                miningField.setUsageType(FieldUsageType.TARGET);
            } else {
                miningField.setUsageType(FieldUsageType.valueOf(params.get("usageType").toString().toUpperCase()));
            }

            miningSchema.withMiningFields(miningField);
        }
        return miningSchema;
    }

}
