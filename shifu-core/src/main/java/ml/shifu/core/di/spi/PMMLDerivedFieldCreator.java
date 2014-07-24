package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.ModelStats;

public interface PMMLDerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params);
}
