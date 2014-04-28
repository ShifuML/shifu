/**
 * Copyright [2012-2014] eBay Software Foundation
 *  
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *  
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.encog.ml.BasicML;


/**
 * 
 * Calculate the score for each evaluation data
 * 
 */
public class EvalScoreUDF extends AbstractTrainerUDF<Tuple> {
    
    private static final String SCHEMA_PREFIX = "shifu::";
    
	private EvalConfig evalConfig;
	private ModelRunner modelRunner;
	private String[] header;
	
	private int modelCnt;

	public EvalScoreUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName) throws IOException {
		super(source, pathModelConfig, pathColumnConfig);
		
		evalConfig = modelConfig.getEvalConfigByName(evalSetName);
		
		if(evalConfig.getModelsPath() != null) {
			//renew columnConfig
			this.columnConfigList = ShifuFileUtils.searchColumnConfig(evalConfig, columnConfigList);
		}
		
		// create model runner
		this.header = CommonUtils.getHeaders(
				evalConfig.getDataSet().getHeaderPath(), 
				evalConfig.getDataSet().getHeaderDelimiter(), 
				evalConfig.getDataSet().getSource());
		
		List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, evalConfig, evalConfig.getDataSet().getSource());
		modelRunner = new ModelRunner(modelConfig, columnConfigList, this.header, evalConfig.getDataSet().getDataDelimiter(), models);
		modelCnt = models.size();
	}

	public Tuple exec(Tuple input) throws IOException {
		CaseScoreResult cs = modelRunner.compute(input);
		if ( cs == null) {
			log.error("Get null result, for input: " + input.toDelimitedString("|"));
			return null;
		}
		
		Tuple tuple = TupleFactory.getInstance().newTuple();
		
		Map<String,String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.header);
		String tag = rawDataMap.get(modelConfig.getTargetColumnName(evalConfig));
		tuple.append(StringUtils.trimToEmpty(tag));
		
		if ( StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName()) ) {
			tuple.append(rawDataMap.get(evalConfig.getDataSet().getWeightColumnName()));
		} else {
			tuple.append(Double.valueOf(1.0));
		}
		
        tuple.append(cs.getAvgScore());
        tuple.append(cs.getMaxScore());
        tuple.append(cs.getMinScore());
        tuple.append(cs.getMedianScore());
		
		for (Integer score : cs.getScores()) {
			tuple.append(score);
		}
				
		// append meta data
		List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
		if ( CollectionUtils.isNotEmpty(metaColumns) ) {
			for (String meta : metaColumns ) {
				tuple.append(rawDataMap.get(meta));
			}
		}
		
		return tuple;
	}
	
	/**
	 * output the schema for evaluation score
	 */
	public Schema outputSchema(Schema input) {
		try {
			Schema tupleSchema = new Schema();
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + modelConfig.getTargetColumnName(evalConfig), DataType.CHARARRAY));
			
			String weightName = StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName())
	                    ? "weight": evalConfig.getDataSet().getWeightColumnName();
			
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + weightName, DataType.CHARARRAY));
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "mean", DataType.INTEGER));
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "max", DataType.INTEGER));
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "min", DataType.INTEGER));
			tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "median", DataType.INTEGER));
			
			for (int i = 0 ; i < modelCnt; i ++) {
				tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "model" + i, DataType.INTEGER));
			}
			
			List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
			if ( CollectionUtils.isNotEmpty(metaColumns) ) {
		        for(String columnName : metaColumns) {
		        	tupleSchema.add(new FieldSchema(columnName, DataType.CHARARRAY));
		        }
		    }
			
			return new Schema(new Schema.FieldSchema("EvalScore", tupleSchema, DataType.TUPLE));
					
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
	}
	
}
