package ml.shifu.core.di.builtin.targets;

import ml.shifu.core.di.spi.PMMLTargetsCreator;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

public class BinaryPMMLTargetsCreator implements PMMLTargetsCreator {

    public Targets create(PMML pmml, Params params) {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOptype(OpType.CATEGORICAL);
        target.setField(new FieldName((String) params.get("targetFieldName")));

        TargetValue pos = new TargetValue();
        pos.setValue((String) params.get("posFieldValue", "P"));
        pos.setDisplayValue((String) params.get("posFieldDisplayValue", "Positive"));

        TargetValue neg = new TargetValue();
        neg.setValue((String) params.get("negFieldValue", "N"));
        neg.setDisplayValue((String) params.get("negFieldDisplayValue", "Negative"));

        target.withTargetValues(pos, neg);

        targets.withTargets(target);


        // Add to PMML
        String modelName = params.get("modelName").toString();
        Model model = PMMLUtils.getModelByName(pmml, modelName);
        model.setTargets(targets);

        return targets;
    }

}
