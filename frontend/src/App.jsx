import { useState } from "react";
import MoleculeStructure from "./MoleculeStructure";
import proteinData from "./proteins.json";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

function App() {
  // ==================================================
  // State
  // ==================================================

  const [proteins, setProteins] = useState([]);
  const [selectedProtein, setSelectedProtein] = useState("");

  const [loadingProteins, setLoadingProteins] = useState(true);
  const [screening, setScreening] = useState(false);

  const [results, setResults] = useState([]);
  const [screenedProtein, setScreenedProtein] = useState("");

  const [viewMode, setViewMode] = useState("all");
  const [sortBy, setSortBy] = useState("fusion");

  const [error, setError] = useState("");

  // ==================================================
  // Load Proteins
  // ==================================================

  useEffect(() => {
    fetch(`${API_URL}/proteins`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load proteins.");
        }

        return response.json();
      })
      .then((data) => {
        setProteteinsSafely(data.proteins);
      })
      .catch((error) => {
        console.error(error);
        setError("Could not load proteins.");
        setLoadingProteins(false);
      });
  }, []);

  function setProteteinsSafely(proteinList) {
    const safeProteinList = Array.isArray(proteinList)
      ? proteinList
      : [];

    setProteins(safeProteinList);

    if (safeProteinList.length > 0) {
      setSelectedProtein(safeProteinList[0]);
    }

    setLoadingProteins(false);
  }

  // ==================================================
  // Run Screening
  // ==================================================

  async function runScreening() {
    if (!selectedProtein) {
      return;
    }

    setScreening(true);
    setError("");
    setResults([]);

    // Reset controls for a new protein
    setViewMode("all");
    setSortBy("fusion");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
      const response = await fetch(
        `${API_URL}/screen/${encodeURIComponent(selectedProtein)}`,
        { signal: controller.signal }
      );

      if (!response.ok) {
        throw new Error("Screening request failed.");
      }

      const data = await response.json();

      setResults(Array.isArray(data.results) ? data.results : []);
      setScreenedProtein(data.protein_id);
    } catch (error) {
      console.error(error);

      if (error.name === "AbortError") {
        setError(
          "The prediction server is still starting. Please try again in a moment."
        );
      } else {
        setError("Screening failed. Please try again in a moment.");
      }
    } finally {
      clearTimeout(timeoutId);
      setScreening(false);
    }
  }

  // ==================================================
  // Filter Results
  // ==================================================

  const filteredResults =
    viewMode === "test"
      ? results.filter((result) => result.split === "test")
      : results;

  // ==================================================
  // Sort Results
  // ==================================================

  const displayedResults = [...filteredResults].sort((a, b) => {
    if (sortBy === "fusion") {
      return b.transformer_pkd - a.transformer_pkd;
    }

    if (sortBy === "xgboost") {
      return b.xgboost_pkd - a.xgboost_pkd;
    }

    if (sortBy === "observed") {
      return b.actual_pkd - a.actual_pkd;
    }

    return 0;
  });

  // ==================================================
  // Page
  // ==================================================

  return (
    <div className="app">
      {/* ==================================================
          Header
      ================================================== */}

      <header className="header">
        <h1>Protein–Ligand Binding Affinity Screening</h1>

        <p>
          Machine learning–based virtual screening using the DAVIS
          drug–target interaction dataset.
        </p>
      </header>

      <main>
        {/* ==================================================
            Project Overview
        ================================================== */}

        <section className="overview-card">
          <p className="section-label">PROJECT OVERVIEW</p>

          <h2>ML-Based Virtual Screening</h2>

          <p className="overview-description">
            This application predicts protein–ligand binding affinity
            (pKd) and ranks candidate compounds for a selected protein
            target using the DAVIS drug–target interaction dataset.
            Predictions are intended to support computational
            prioritization of candidate compounds for downstream
            experimental validation.
          </p>

          {/* ==================================================
              Model Cards
          ================================================== */}

          <div className="model-grid">
            {/* Fusion MLP */}

            <div className="model-card">
              <p className="model-number">MODEL 1</p>

              <h3>Fusion MLP</h3>

              <p>
                <strong>Protein:</strong>{" "}
                ESM-2 35M embedding (480d)
              </p>

              <p>
                <strong>Compound:</strong>{" "}
                ChemBERTa embedding (768d)
              </p>

              <div className="model-flow">
                ESM-2 + ChemBERTa → Fusion MLP → Predicted pKd
              </div>
            </div>

            {/* XGBoost */}

            <div className="model-card">
              <p className="model-number">MODEL 2</p>

              <h3>XGBoost Baseline</h3>

              <p>
                <strong>Protein:</strong>{" "}
                ESM-2 35M embedding (480d)
              </p>

              <p>
                <strong>Compound:</strong>{" "}
                Morgan fingerprint (1024d)
              </p>

              <div className="model-flow">
                ESM-2 + Morgan → XGBoost → Predicted pKd
              </div>
            </div>
          </div>

          {/* ==================================================
              How to Read the Results
          ================================================== */}

          <div className="results-guide">
            <p className="guide-label">HOW TO READ THE RESULTS</p>

            <h3>Prediction vs. Experimental Binding Affinity</h3>

            <p className="guide-intro">
              Both models predict the pKd of the same
              protein–compound pair. The predicted values are
              compared with the experimentally observed pKd
              reported in DAVIS.
            </p>

            <div className="guide-grid">
              {/* Fusion */}

              <div className="guide-item">
                <strong>Fusion MLP pKd</strong>

                <p>
                  Prediction from the primary neural network model
                  using ESM-2 protein embeddings and ChemBERTa
                  compound embeddings.
                </p>
              </div>

              {/* XGBoost */}

              <div className="guide-item">
                <strong>XGBoost pKd</strong>

                <p>
                  Prediction from the baseline model using ESM-2
                  protein embeddings and Morgan molecular
                  fingerprints.
                </p>
              </div>

              {/* Observed */}

              <div className="guide-item">
                <strong>Observed pKd</strong>

                <p>
                  Experimental binding affinity reported in DAVIS
                  and used as the reference value. Higher pKd
                  indicates stronger binding.
                </p>
              </div>
            </div>

            {/* ==================================================
                Split Explanation
            ================================================== */}

            <div className="split-guide">
              <h4>What do Train, Validation, and Test mean?</h4>

              <div className="split-guide-row">
                <span className="split-badge train">
                  Train
                </span>

                <p>
                  This protein–compound pair was used to train
                  the models.
                </p>
              </div>

              <div className="split-guide-row">
                <span className="split-badge validation">
                  Validation
                </span>

                <p>
                  This pair was used during model development
                  for validation and model selection.
                </p>
              </div>

              <div className="split-guide-row">
                <span className="split-badge test">
                  Test
                </span>

                <p>
                  This pair was held out from training and model
                  selection and used for final model evaluation.
                </p>
              </div>
            </div>

            {/* ==================================================
                Evaluation Note
            ================================================== */}

            <div className="evaluation-note">
              <strong>Evaluation note</strong>

              <p>
                The dataset uses a random protein–compound pair
                split. A test pair was excluded from model
                training, although its individual protein or
                compound may appear in other training pairs.
                Test results therefore evaluate held-out pair
                prediction rather than performance on completely
                unseen proteins or compounds.
              </p>
            </div>
          </div>
        </section>

        {/* ==================================================
            Target Protein
        ================================================== */}

        <section className="target-card">
          <p className="section-label">TARGET PROTEIN</p>

          <h2>Select a Protein</h2>

          {loadingProteins ? (
            <p>Connecting to prediction server... First Load may take up to a minute.</p>
          ) : (
            <select
              value={selectedProtein}
              onChange={(event) =>
                setSelectedProtein(event.target.value)
              }
            >
              {proteins.map((protein) => (
                <option
                  key={protein}
                  value={protein}
                >
                  {protein}
                </option>
              ))}
            </select>
          )}

          <div className="selected-protein">
            Selected target:
            <strong>
              {" "}
              {selectedProtein}
            </strong>
          </div>

          <button
            onClick={runScreening}
            disabled={
              screening ||
              loadingProteins ||
              !selectedProtein
            }
          >
            {screening
              ? "Starting prediction server & running screening..."
              : "Run Screening"}
          </button>

          {screening && (
            <p className="screening-status">
              First request may take a few minutes.
            </p>
          )}

          {error && (
            <p className="error">
              {error}
            </p>
          )}
        </section>

        {/* ==================================================
            Screening Results
        ================================================== */}

        {results.length > 0 && (
          <section className="results-section">
            {/* ==================================================
                Results Header
            ================================================== */}

            <div className="results-header">
              <div>
                <p className="section-label">
                  SCREENING RESULTS
                </p>

                <h2>{screenedProtein}</h2>

                {/* ------------------------------------------
                    All Pairs / Test Only
                ------------------------------------------ */}

                <div className="view-toggle">
                  <button
                    className={
                      viewMode === "all"
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setViewMode("all")
                    }
                  >
                    All Pairs
                  </button>

                  <button
                    className={
                      viewMode === "test"
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setViewMode("test")
                    }
                  >
                    Test Only
                  </button>
                </div>

                {/* ------------------------------------------
                    View Description
                ------------------------------------------ */}

                <p className="view-description">
                  {viewMode === "all"
                    ? "Explore all DAVIS compounds for this target. Split labels show whether each protein–compound pair belongs to the training, validation, or test set."
                    : "Showing only held-out test pairs used for final model evaluation."}
                </p>

                {/* ------------------------------------------
                    Sorting
                ------------------------------------------ */}

                <div className="sort-control">
                  <label htmlFor="sort-results">
                    Sort by
                  </label>

                  <select
                    id="sort-results"
                    value={sortBy}
                    onChange={(event) =>
                      setSortBy(event.target.value)
                    }
                  >
                    <option value="fusion">
                      Fusion MLP pKd — Highest First
                    </option>

                    <option value="xgboost">
                      XGBoost pKd — Highest First
                    </option>

                    <option value="observed">
                      Observed pKd — Highest First
                    </option>
                  </select>
                </div>
              </div>

              <p>
                <strong>
                  {displayedResults.length}
                </strong>
                {" "}of{" "}
                {results.length} compounds
              </p>
            </div>

            {/* ==================================================
                Overall Model Performance
            ================================================== */}

            <div className="performance-section">
              <div className="performance-header">
                <p className="section-label">
                  MODEL PERFORMANCE
                </p>

                <h3>
                  Held-Out Test Set
                </h3>

                <p>
                  Overall performance across all held-out
                  protein–compound test pairs.
                </p>
              </div>

              <div className="performance-grid">
                {/* ------------------------------------------
                    Fusion Performance
                ------------------------------------------ */}

                <div className="performance-card">
                  <p className="performance-model-label">
                    PRIMARY MODEL
                  </p>

                  <h3>Fusion MLP</h3>

                  <div className="metric-row">
                    <span>RMSE</span>
                    <strong>0.5222</strong>
                  </div>

                  <div className="metric-row">
                    <span>R²</span>
                    <strong>0.6051</strong>
                  </div>

                  <div className="metric-row">
                    <span>Pearson r</span>
                    <strong>0.7780</strong>
                  </div>
                </div>

                {/* ------------------------------------------
                    XGBoost Performance
                ------------------------------------------ */}

                <div className="performance-card">
                  <p className="performance-model-label">
                    BASELINE
                  </p>

                  <h3>XGBoost</h3>

                  <div className="metric-row">
                    <span>RMSE</span>
                    <strong>0.5423</strong>
                  </div>

                  <div className="metric-row">
                    <span>R²</span>
                    <strong>0.5740</strong>
                  </div>

                  <div className="metric-row">
                    <span>Pearson r</span>
                    <strong>0.7593</strong>
                  </div>
                </div>
              </div>

              <p className="performance-note">
                Metrics are calculated on the full held-out
                test set, not only the compounds displayed for
                the currently selected protein.
              </p>
            </div>

            {/* ==================================================
                Compound Cards
            ================================================== */}

            <div className="compound-grid">
              {displayedResults.map((result) => {
                // ------------------------------------------
                // Absolute prediction errors
                // ------------------------------------------

                const fusionError = Math.abs(
                  result.transformer_pkd -
                  result.actual_pkd
                );

                const xgboostError = Math.abs(
                  result.xgboost_pkd -
                  result.actual_pkd
                );

                return (
                  <div
                    className="compound-card"
                    key={result.drug_id}
                  >
                    {/* ------------------------------------------
                        Molecule Structure
                    ------------------------------------------ */}

                    <div className="molecule-container">
                      <MoleculeStructure
                        smiles={result.smiles}
                      />
                    </div>

                    {/* ------------------------------------------
                        Compound Information
                    ------------------------------------------ */}

                    <div className="compound-header">
                      <div>
                        <p className="compound-label">
                          COMPOUND
                        </p>

                        <h3>
                          {result.drug_id}
                        </h3>
                      </div>

                      <span
                        className={
                          `split-badge ${result.split}`
                        }
                      >
                        {result.split}
                      </span>
                    </div>

                    {/* ------------------------------------------
                        Prediction Results
                    ------------------------------------------ */}

                    <div className="prediction-list">
                      {/* Fusion MLP */}

                      <div className="prediction-row">
                        <span>
                          Fusion MLP
                        </span>

                        <strong>
                          {result.transformer_pkd.toFixed(3)}
                        </strong>
                      </div>

                      <div className="error-row">
                        <span>
                          Absolute error
                        </span>

                        <span>
                          {fusionError.toFixed(3)}
                        </span>
                      </div>

                      {/* XGBoost */}

                      <div className="prediction-row">
                        <span>
                          XGBoost
                        </span>

                        <strong>
                          {result.xgboost_pkd.toFixed(3)}
                        </strong>
                      </div>

                      <div className="error-row">
                        <span>
                          Absolute error
                        </span>

                        <span>
                          {xgboostError.toFixed(3)}
                        </span>
                      </div>

                      {/* Observed */}

                      <div className="prediction-row observed">
                        <span>
                          Observed pKd
                        </span>

                        <strong>
                          {result.actual_pkd.toFixed(3)}
                        </strong>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;