/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.pig;

import org.apache.pig.piggybank.storage.CSVExcelStorage;
import org.apache.pig.builtin.PigStorage;
import org.apache.pig.data.Tuple;
import org.apache.pig.ResourceSchema;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.pig.impl.logicalLayer.FrontendException;
import org.apache.pig.backend.hadoop.executionengine.mapReduceLayer.PigSplit;

import java.io.IOException;
import java.util.List;


public class ShifuPigStorage extends PigStorage {

    private PigStorage shifuStorage;

    public ShifuPigStorage(String isCSV) {
        if("true".equals(isCSV.toLowerCase()) && isCSV != null) {
           shifuStorage = new CSVExcelStorage("\t", "NO_MULTILINE", "UNIX", "WRITE_OUTPUT_HEADER");
        } else {
           shifuStorage = new PigStorage("\t", "-schema");
        }
    }
    
    public ShifuPigStorage(String isCSV, String delimiter) {
        super(delimiter);
        if("true".equals(isCSV.toLowerCase()) && isCSV != null) {
           shifuStorage = new CSVExcelStorage(delimiter, "NO_MULTILINE", "UNIX", "WRITE_OUTPUT_HEADER");
        } else {
           shifuStorage = new PigStorage(delimiter, "-schema");
        }
    }

    @Override
    public void putNext(Tuple tupleToWrite) throws IOException {
        shifuStorage.putNext(tupleToWrite);
    }

    @Override
    public Tuple getNext() throws IOException { 
        return shifuStorage.getNext();
    }

    @Override
    public void setLocation(String location, Job job) throws IOException {
        shifuStorage.setLocation(location, job);
    }

    @SuppressWarnings("rawtypes")
    @Override
    public InputFormat getInputFormat() {
        return shifuStorage.getInputFormat();
    }

    @Override
    public void prepareToRead(@SuppressWarnings("rawtypes") RecordReader reader, PigSplit split) {
        shifuStorage.prepareToRead(reader, split);
    }

    @Override
    public RequiredFieldResponse pushProjection(RequiredFieldList requiredFieldList) throws FrontendException {
        return shifuStorage.pushProjection(requiredFieldList);
    }

    @Override
    public void setUDFContextSignature(String signature) {
        shifuStorage.setUDFContextSignature(signature);
    }

    @Override
    public void setStoreFuncUDFContextSignature(String signature) {
        shifuStorage.setStoreFuncUDFContextSignature(signature);
    }

    @Override
    public List<OperatorSet> getFeatures() {
        return shifuStorage.getFeatures();
    }
    
    @SuppressWarnings("rawtypes")
    @Override
    public void prepareToWrite(RecordWriter writer) {
        shifuStorage.prepareToWrite(writer);
    }

    @Override
    public void checkSchema(ResourceSchema s) throws IOException { 
        shifuStorage.checkSchema(s);
    }

    @Override
    public void storeSchema(ResourceSchema schema, String location,
        Job job) throws IOException {
        shifuStorage.storeSchema(schema, location, job);
    }

    @Override
    public ResourceSchema getSchema(String location,
        Job job) throws IOException {
        return shifuStorage.getSchema(location, job);
    }
    
    @SuppressWarnings("rawtypes")
    @Override
    public OutputFormat getOutputFormat() {
        return shifuStorage.getOutputFormat();
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        shifuStorage.setStoreLocation(location, job);
    }
}
