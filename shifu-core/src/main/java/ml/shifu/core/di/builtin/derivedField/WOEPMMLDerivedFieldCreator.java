package ml.shifu.core.di.builtin.derivedfield;

import ml.shifu.core.di.spi.PMMLDerivedFieldCreator;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

import java.util.List;

public class WOEPMMLDerivedFieldCreator implements PMMLDerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {


        DerivedField derivedField = new DerivedField();
        derivedField.setName(new FieldName(dataField.getName().getValue() + "_transformed"));
        derivedField.setOptype(dataField.getOptype());
        derivedField.setDataType(dataField.getDataType());

        UnivariateStats univariateStats = PMMLUtils.getUnivariateStatsByFieldName(modelStats, dataField.getName());

        List<Extension> extensions = univariateStats.getContStats().getExtensions();

        List<Double> woe = CommonUtils.stringToDoubleList(PMMLUtils.getExtension(extensions, "BinWOE").getValue());


        Discretize discretize = new Discretize();
        discretize.setField(dataField.getName());

        int size = univariateStats.getContStats().getIntervals().size();
        for (int i = 0; i < size; i++) {
            DiscretizeBin bin = new DiscretizeBin();
            bin.setInterval(univariateStats.getContStats().getIntervals().get(i));
            bin.setBinValue(woe.get(i).toString());
            discretize.withDiscretizeBins(bin);
        }

        derivedField.setExpression(discretize);

        return derivedField;
    }
}
