package ml.shifu.shifu.di.spi;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.ModelStats;

public interface DerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats);
}
