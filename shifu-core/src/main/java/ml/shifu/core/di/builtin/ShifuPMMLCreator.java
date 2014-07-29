package ml.shifu.core.di.builtin;

import ml.shifu.core.di.spi.PMMLCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.PMML;


public class ShifuPMMLCreator implements PMMLCreator {

    public PMML create(Params params) {


        PMML pmml = new PMML();


        //TODO: add Shifu info to Header


        return pmml;
    }
}
