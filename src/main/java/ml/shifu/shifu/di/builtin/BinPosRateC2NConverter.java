package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;

public class BinPosRateC2NConverter {

    private ColumnBinningResult columnBinningResult;

    public void setColumnBinningResult(ColumnBinningResult columnBinningResult) {
        this.columnBinningResult = columnBinningResult;
    }

    public NumericalValueObject convert(CategoricalValueObject cvo) {
        if (columnBinningResult == null) {
            throw new RuntimeException("No ColumnBinningResult specified. Call setColumnBinningResult() first.");
        }
        int index = columnBinningResult.getBinCategory().indexOf(cvo.getValue());

        NumericalValueObject nvo = new NumericalValueObject();

        if (index == -1) {
            // TODO: how to deal with missing data
            nvo.setValue(0.0);
        } else {
            nvo.setValue(columnBinningResult.getBinPosRate().get(index));
        }
        nvo.setWeight(cvo.getWeight());


        return nvo;
    }
}
