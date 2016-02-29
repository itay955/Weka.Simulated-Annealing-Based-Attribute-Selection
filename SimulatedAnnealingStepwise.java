/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


package weka.attributeSelection;

import weka.core.*;

import java.util.*;

/**
 * <!-- globalinfo-start --> SimulatedAnnealingStepwise :<br/>
 * <br/>
 * The main idea is that in each running iteration we are starting with a random set
 * of features and until the average change is below a threshold we are generating a
 * random attribute to change the current permutation. If it in the set we try to
 * see what will happen if we remove it. If it out of the set we try to see what
 * will happen if we will remove it. If the change enlarges the merit that we are
 * good with the change. If the change is for the bad it is dependent with the
 * current temperature so as long as we traverse the probability of making a bad
 * change is reduced.  Finally after the iterations are done we are producing the
 * subset with the best merit among all.
 * References:
 * Kuhn and Johnson (2013), Applied Predictive Modeling, Springer
 * Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983).
 * Optimization by simulated annealing. Science, 220(4598), 671.
 * <br/>
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <p/>
 * <!-- options-start --> Valid options are:
 * <p/>
 * <p/>
 * <pre>
 * -C
 *  Use conservative  search
 * </pre>
 * <p/>
 * <p/>
 * <pre>
 * -P &lt;start set&gt;
 *  Specify a starting set of attributes.
 *  Eg. 1,3,5-7.
 * </pre>
 * <p/>
 * <pre>
 * -R
 *  Specify the random seed
 * </pre>
 * <p/>
 * <pre>
 * -T &lt;temperature&gt;
 *  Specify annealing start temperature
 * </pre>
 * <p/>
 * <pre>
 * -A &lt;coefficient&gt;
 *  Specify annealing coefficient
 * </pre>
 * <p/>
 * <pre>
 * -S &lt;coefficient&gt;
 *  Specify stopping threshold
 * </pre>
 * <p/>
 * <pre>
 * -I &lt;number of iterations&gt;
 *  Specify number of iterations to start
 * </pre>
 * <p/>
 * <pre>
 * -D
 *  Print debugging output
 * </pre>
 * <p/>
 * <!-- options-end -->
 *
 * @author Itay Hazan (itayhaz@post.bgu.ac.il)
 * @author Andrey Finkelstein (andreyfi@post.bgu.ac.il)
 * @version $Revision: 1 $
 * this implementation is done by converting the GreedyStepwise search method implemented by Mark Hall
 * References:
 * Kuhn and Johnson (2013), Applied Predictive Modeling, Springer
 * Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983).
 * Optimization by simulated annealing. Science, 220(4598), 671.
 */
public class SimulatedAnnealingStepwise extends ASSearch implements
        StartSetHandler, OptionHandler {

    /**
     * does the data have a class
     */
    protected boolean m_hasClass;

    /**
     * holds the class index
     */
    protected int m_classIndex;

    /**
     * number of attributes in the data
     */
    protected int m_numAttribs;

    /**
     * the merit of the best subset found
     */
    protected double m_bestMerit;

    /**
     * the best subset found
     */
    protected BitSet m_best_group;

    /**
     * the evaluator
     */
    protected ASEvaluation m_ASEval;

    /**
     * number of instances in the data
     */
    protected Instances m_Instances;

    /**
     * holds the start set for the search as a Range
     */
    protected Range m_startRange;

    /**
     * holds an array of starting attributes
     */
    protected int[] m_starting;

    /**
     * If set then attributes will continue to be added during a  search as
     * long as the merit does not degrade
     */
    protected boolean m_conservativeSelection = false;

    /**
     * Print debugging output
     */
    protected boolean m_debug = false;

    //--------------------------------------------------Simulated Variables ------------------------------------------------
    //--------------------------------------------------Simulated Variables ------------------------------------------------
    //--------------------------------------------------Simulated Variables ------------------------------------------------
    /**
     * random generator
     */
    protected Random random;
    /**
     * starting temperature in each iteration
     */
    protected double annealing_temperature = 0.1;
    /**
     * the coefficient that updating the temperature in each iteration
     */
    protected double annealing_coefficient = 0.4;
    /**
     * the threshold that determines if the iteration has converged
     */
    protected double annealing_change_threshold = 0.0005;
    /**
     * the random starting seed
     */
    protected int annealing_random_seed = 1;
    /**
     * the number of iterations to start the process, finally the algorithm pick the best one
     */
    protected int annealing_iterations = 5;
    /**
     * the minimum steps in each iteration
     */
    private int minimum_steps = 10;

    /**
     * Constructor
     */
    public SimulatedAnnealingStepwise() {
        m_startRange = new Range();
        m_starting = null;
        resetOptions();
    }

    /**
     *
     * @return the best merit among all iterations
     */
    public double getM_bestMerit() {
        return m_bestMerit;
    }

    /**
     * Returns a string describing this search method
     *
     * @return a description of the search suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "SimulatedAnnealingStepwise :\n\n"
                +"The main idea is that in each running iteration we are starting with a random set "
                +"of features and until the average change is below a threshold we are generating a "
                +"random attribute to change the current permutation. If it in the set we try to "
                +"see what will happen if we remove it. If it out of the set we try to see what "
                +"will happen if we will remove it. If the change enlarges the merit that we are "
                +"good with the change. If the change is for the bad it is dependent with the "
                +"current temperature so as long as we traverse the probability of making a bad "
                +"change is reduced.  Finally after the iterations are done we are producing the "
                +"subset with the best merit among all. References:"
                +"Kuhn and Johnson (2013), Applied Predictive Modeling, Springer"
                +"Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983)."
                +"Optimization by simulated annealing. Science, 220(4598), 671."
                ;
    }

    /**
     *
     * @return the total number of iterations
     */
    public int getAnnealing_iterations() {
        return annealing_iterations;
    }

    /**
     * sets the total amount of iterations
     * @param annealing_iterations
     */
    public void setAnnealing_iterations(int annealing_iterations) {
        this.annealing_iterations = annealing_iterations;
    }

    /**
     *
     * @return the random seed
     */
    public int getAnnealing_random_seed() {
        return annealing_random_seed;
    }

    /**
     * sets the random generating seed
     * @param annealing_random_seed
     */
    public void setAnnealing_random_seed(int annealing_random_seed) {
        this.annealing_random_seed = annealing_random_seed;
    }

    /**
     *
     * @return the change threshold that stops the running and defined the convergence
     */
    public double getAnnealing_change_threshold() {
        return annealing_change_threshold;
    }

    /**
     * sets the change threshold that stops the running and defined the convergence
     * @param annealing_change_threshold
     */
    public void setAnnealing_change_threshold(double annealing_change_threshold) {
        this.annealing_change_threshold = annealing_change_threshold;
    }

    /**
     *
     * @return the annealing coefficient that updates the current temperature each iteration
     */
    public double getAnnealing_coefficient() {
        return annealing_coefficient;
    }

    /**
     * sets the annealing coefficient that updates the current temperature each iteration
     * @param annealing_coefficient
     */
    public void setAnnealing_coefficient(double annealing_coefficient) {
        this.annealing_coefficient = annealing_coefficient;
    }

    /**
     *
     * @return the current temperature of the annealing process
     */
    public double getAnnealing_temperature() {
        return annealing_temperature;
    }

    /**
     * sets the annealing temperature used at the beginning of each iteration
     * @param annealing_temperature
     */
    public void setAnnealing_temperature(double annealing_temperature) {
        this.annealing_temperature = annealing_temperature;
    }


    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String startSetTipText() {
        return "Set the start point for the search. This is specified as a comma "
                + "seperated list off attribute indexes starting at 1. It can include "
                + "ranges. Eg. 1,2,5-9,17.";
    }

    /**
     * Returns the starting set, list of attributes (and or attribute ranges) as a String
     *
     * @return a list of attributes (and or attribute ranges)
     */
    @Override
    public String getStartSet() {
        return m_startRange.getRanges();
    }

    /**
     * Sets a starting set of attributes for the search. It is the search method's
     * responsibility to report this start set (if any) in its toString() method.
     *
     * @param startSet a string containing a list of attributes (and or ranges) eg. 1,2,6,10-15.
     * @throws Exception if start set can't be set.
     */
    @Override
    public void setStartSet(String startSet) throws Exception {
        m_startRange.setRanges(startSet);
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String conservativeSelectionTipText() {
        return "If true then attributes will continue to be added to "
                + "the best subset as long as merit does not degrade.";
    }

    /**
     * Gets whether conservative selection has been enabled
     * @return true if conservative selection is enabled
     */
    public boolean getConservativeSelection() {
        return m_conservativeSelection;
    }

    /**
     * Set whether attributes should continue to be added during search
     * as long as merit does not decrease
     *
     * @param c true if attributes should continue to be added
     */
    public void setConservativeSelection(boolean c) {
        m_conservativeSelection = c;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String debuggingOutputTipText() {
        return "Output debugging information to the console";
    }

    /**
     * Get whether to output debugging info to the console
     *
     * @return true if debugging info is to be output
     */
    public boolean getDebuggingOutput() {
        return m_debug;
    }

    /**
     * Set whether to output debugging info to the console
     *
     * @param d true if debugging info is to be output
     */
    public void setDebuggingOutput(boolean d) {
        m_debug = d;
    }


    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     **/
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(8);

        newVector.addElement(new Option("\tUse conservative  search", "-C", 0, "-C"));
        newVector.addElement(new Option("\tPrint debugging output", "-D", 0, "-D"));
        newVector.addElement(new Option("\tSpecify a starting set of attributes." + "\n\tEg. 1,3,5-7.", "-P", 1, "-P <start set>"));
        newVector.addElement(new Option("\tSpecify number of iterations to start", "-I", 1, "<Integer>"));
        newVector.addElement(new Option("\tSpecify annealing start temperature", "-T", 1, "<0-1>"));
        newVector.addElement(new Option("\tSpecify annealing coefficient ", "-A", 1, "<0-1>"));
        newVector.addElement(new Option("\tSpecify the random seed", "-R", 1, "<Integer>"));
        newVector.addElement(new Option("\tSpecify stopping threshold", "-S", 1, "<0-1>"));

        return newVector.elements();

    }

    /**
     * Gets the current settings of ReliefFAttributeEval.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {

//        annealing_temperature = 0.1;
//        annealing_coefficient = 0.4;
//        annealing_change_threshold = 0.0005;
//        annealing_iterations = 5;
//        annealing_random_seed = 120;

        Vector<String> options = new Vector<String>();

        if (getConservativeSelection()) {
            options.add("-C");
        }
        if (getDebuggingOutput()) {
            options.add("-D");
        }

        if (!(getStartSet().equals(""))) {
            options.add("-P");
            options.add("" + startSetToString());
        }

        options.add("-R");
        options.add("" + getAnnealing_random_seed());
        options.add("-T");
        options.add("" + getAnnealing_temperature());
        options.add("-I");
        options.add("" + getAnnealing_iterations());
        options.add("-A");
        options.add("" + getAnnealing_coefficient());
        options.add("-S");
        options.add("" + getAnnealing_change_threshold());

        return options.toArray(new String[0]);
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString;
        resetOptions();

        setConservativeSelection(Utils.getFlag('C', options));
        setDebuggingOutput(Utils.getFlag('D', options));

        optionString = Utils.getOption('P', options);
        if (optionString.length() != 0) {
            setStartSet(optionString);
        }

        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            Double temp = Double.valueOf(optionString);
            setAnnealing_temperature(temp.doubleValue());
        }

        optionString = Utils.getOption('A', options);
        if (optionString.length() != 0) {
            Double temp = Double.valueOf(optionString);
            setAnnealing_coefficient(temp.doubleValue());
        }

        optionString = Utils.getOption('S', options);
        if (optionString.length() != 0) {
            Double temp = Double.valueOf(optionString);
            setAnnealing_change_threshold(temp.doubleValue());
        }

        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            setAnnealing_iterations(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption('R', options);
        if (optionString.length() != 0) {
            setAnnealing_random_seed(Integer.parseInt(optionString));
        }
    }

    /**
     * converts the array of starting attributes to a string. This is used by
     * getOptions to return the actual attributes specified as the starting set.
     * This is better than using m_startRanges.getRanges() as the same start set
     * can be specified in different ways from the command line---eg 1,2,3 == 1-3.
     * This is to ensure that stuff that is stored in a database is comparable.
     *
     * @return a comma separated list of individual attribute numbers as a String
     */
    protected String startSetToString() {
        StringBuffer FString = new StringBuffer();
        boolean didPrint;

        if (m_starting == null) {
            return getStartSet();
        }
        for (int i = 0; i < m_starting.length; i++) {
            didPrint = false;

            if ((m_hasClass == false) || (m_hasClass == true && i != m_classIndex)) {
                FString.append((m_starting[i] + 1));
                didPrint = true;
            }

            if (i == (m_starting.length - 1)) {
                FString.append("");
            } else {
                if (didPrint) {
                    FString.append(",");
                }
            }
        }

        return FString.toString();
    }

    /**
     *
     * @return a description of the search as a String.
     */
    @Override
    public String toString() {
        StringBuffer FString = new StringBuffer();
        FString.append("\tSimulated annealing ("
                + ".\n\tStart set: ");

        if (m_starting == null) {
            FString.append("random set\n");

        } else {
            FString.append(startSetToString() + "\n");
        }

        return FString.toString();
    }

    /**
     * Searches the attribute subset space by  selection.
     *
     * @param ASEval the attribute evaluator to guide the search
     * @param data   the training instances.
     * @return an array (not necessarily ordered) of selected attribute indexes
     * @throws Exception if the search can't be completed
     */
    @Override
    public int[] search(ASEvaluation ASEval, Instances data) throws Exception {

        if (data != null) { // this is a fresh run so reset
            resetOptions();
            m_Instances = new Instances(data, 0);
        }
        m_ASEval = ASEval;
        m_numAttribs = m_Instances.numAttributes();

        if (!(m_ASEval instanceof SubsetEvaluator)) {
            throw new Exception(m_ASEval.getClass().getName() + " is not a " + "Subset evaluator!");
        }

        m_startRange.setUpper(m_numAttribs - 1);
        if (!(getStartSet().equals(""))) {
            m_starting = m_startRange.getSelection();
        }

        if (m_ASEval instanceof UnsupervisedSubsetEvaluator) {
            m_hasClass = false;
            m_classIndex = -1;
        } else {
            m_hasClass = true;
            m_classIndex = m_Instances.classIndex();
        }

        final SubsetEvaluator ASEvaluator = (SubsetEvaluator) m_ASEval;


        int number_of_iterations = 0;
        while (number_of_iterations < annealing_iterations) {

            int i;
            BitSet current_best_group = new BitSet(m_numAttribs);
            BitSet temp_group;
            double best_merit;
            double temp_merit;
            boolean done = false;
            boolean addone = false;
            boolean realImprove;
            boolean annealingMistake;
            double sumMeritChange = 0;
            int stepCounter = 0;
            double iteration_temperature = annealing_temperature;

            // If a starting subset has been supplied, then initialise the bitset
            if (m_starting != null) {
                for (i = 0; i < m_starting.length; i++) {
                    if ((m_starting[i]) != m_classIndex) {
                        current_best_group.set(m_starting[i]);
                    }
                }
            } else {
                current_best_group = selectRandomGroup(m_numAttribs);
            }

            // Evaluate the initial subset
            best_merit = ASEvaluator.evaluateSubset(current_best_group);

            while (!done) {
                temp_group = (BitSet) current_best_group.clone();
                addone = false;
                stepCounter += 1;
                i = m_classIndex;
                while (i == m_classIndex) {
                    i = random.nextInt(m_numAttribs);
                }

                boolean feature_inside_set = temp_group.get(i);

                // set/unset the bit
                if (feature_inside_set) {
                    temp_group.clear(i);
                } else {
                    temp_group.set(i);
                }
                temp_merit = ASEvaluator.evaluateSubset(temp_group);
                if (m_conservativeSelection) {
                    realImprove = (temp_merit >= best_merit);
                } else {
                    realImprove = (temp_merit > best_merit);
                }

                //---------------------------------Annealing Function ---------------------------------/
                //---------------------------------Annealing Function ---------------------------------/
                //---------------------------------Annealing Function ---------------------------------/
                double differential = temp_merit - best_merit;
                annealingMistake = random.nextDouble() <= Math.exp(differential / iteration_temperature);
                iteration_temperature *= annealing_coefficient;

                if (realImprove || annealingMistake) { //should use the new subset
                    addone = true;
                    sumMeritChange += Math.abs(differential);
                }

                done = (sumMeritChange / stepCounter < annealing_change_threshold) && stepCounter > minimum_steps;
                if (addone) {
                    if (feature_inside_set) {
                        current_best_group.clear(i);
                    } else {
                        current_best_group.set(i);
                    }
                    best_merit = temp_merit;
                    if (m_debug) {
                        System.err.print("Current subset is: ");
                        int[] atts = attributeList(current_best_group);
                        for (int a : atts) {
                            System.err.print("" + (a + 1) + " ");
                        }
                        System.err.println("\nMerit: " + best_merit);
                    }
                }
            }

            if (best_merit > m_bestMerit) {
                m_bestMerit = best_merit;
                m_best_group = (BitSet) current_best_group.clone();
            }
            number_of_iterations++;


        }
        return attributeList(m_best_group);
    }


    private BitSet selectRandomGroup(int numAtt) {
        BitSet feats = new BitSet(numAtt);
        int numOfFeaturesToStart = (int) Math.sqrt(random.nextInt(numAtt));
        while (numOfFeaturesToStart > 0) {
            int currFeat = random.nextInt(numAtt);
            if (!feats.get(currFeat) && currFeat != m_classIndex) {
                feats.set(currFeat);
                numOfFeaturesToStart--;
            }
        }
        return feats;
    }


    /**
     * converts a BitSet into a list of attribute indexes
     *
     * @param group the BitSet to convert
     * @return an array of attribute indexes
     **/
    protected int[] attributeList(BitSet group) {
        int count = 0;

        // count how many were selected
        for (int i = 0; i < m_numAttribs; i++) {
            if (group.get(i)) {
                count++;
            }
        }

        int[] list = new int[count];
        count = 0;

        for (int i = 0; i < m_numAttribs; i++) {
            if (group.get(i)) {
                list[count++] = i;
            }
        }

        return list;
    }

    /**
     * Resets options
     */
    protected void resetOptions() {
        m_best_group = null;
        m_ASEval = null;
        m_Instances = null;
        random = new Random(annealing_random_seed);
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }
}
