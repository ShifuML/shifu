package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;

public interface C2NConverter {

    NumericalValueObject convert(CategoricalValueObject cvo);
}
