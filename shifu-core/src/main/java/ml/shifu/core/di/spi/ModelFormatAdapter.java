package ml.shifu.core.di.spi;

import org.dmg.pmml.PMML;

public interface ModelFormatAdapter {

    public Object exec(Object originModel);



}
