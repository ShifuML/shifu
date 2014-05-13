package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;

public interface N2CConverter {

    CategoricalValueObject convert(NumericalValueObject nvo);
}
