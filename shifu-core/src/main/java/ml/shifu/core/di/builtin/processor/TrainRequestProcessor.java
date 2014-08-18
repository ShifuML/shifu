package ml.shifu.core.di.builtin.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.TrainingService;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.*;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class TrainRequestProcessor implements RequestProcessor {

    private static final Logger log = LoggerFactory.getLogger(TrainRequestProcessor.class);

    public void exec(Request req) throws Exception {


    }
}
