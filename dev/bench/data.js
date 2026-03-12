window.BENCHMARK_DATA = {
  "lastUpdate": 1773359915063,
  "repoUrl": "https://github.com/xorq-labs/xorq-gallery",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "committer": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "id": "85d7f13f8cf35f90e4db0f2f07f9ebc70fd97d24",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/85d7f13f8cf35f90e4db0f2f07f9ebc70fd97d24"
        },
        "date": 1773359914494,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 5364.7621813944015,
            "unit": "iter/sec",
            "range": "stddev: 0.00033021453288001416",
            "extra": "mean: 186.40155261087108 usec\nrounds: 1283"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4669.7960087921565,
            "unit": "iter/sec",
            "range": "stddev: 0.000361299062776527",
            "extra": "mean: 214.14211629742053 usec\nrounds: 2528"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2328.0965106187964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475486930776509",
            "extra": "mean: 429.5354575890005 usec\nrounds: 1344"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2057.97450668709,
            "unit": "iter/sec",
            "range": "stddev: 0.0007677202376055774",
            "extra": "mean: 485.9146684036391 usec\nrounds: 2111"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2040.0693795887619,
            "unit": "iter/sec",
            "range": "stddev: 0.0010777750905628773",
            "extra": "mean: 490.1794076246468 usec\nrounds: 2046"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2309.837895563966,
            "unit": "iter/sec",
            "range": "stddev: 0.00005107826294422144",
            "extra": "mean: 432.9308138551609 usec\nrounds: 1891"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2302.2794298732397,
            "unit": "iter/sec",
            "range": "stddev: 0.0011561667516904654",
            "extra": "mean: 434.3521411973258 usec\nrounds: 2238"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2364.780153812327,
            "unit": "iter/sec",
            "range": "stddev: 0.00004755050526112874",
            "extra": "mean: 422.8722904274516 usec\nrounds: 2152"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2133.226959483971,
            "unit": "iter/sec",
            "range": "stddev: 0.0017260213540315186",
            "extra": "mean: 468.77337432576826 usec\nrounds: 1854"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2352.543034604137,
            "unit": "iter/sec",
            "range": "stddev: 0.00004644723320035439",
            "extra": "mean: 425.0719265453396 usec\nrounds: 1974"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2342.8776892934034,
            "unit": "iter/sec",
            "range": "stddev: 0.00005505665762147324",
            "extra": "mean: 426.8255251095047 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2123.02663179036,
            "unit": "iter/sec",
            "range": "stddev: 0.002060974110724872",
            "extra": "mean: 471.02565037382243 usec\nrounds: 2005"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2360.4433380318337,
            "unit": "iter/sec",
            "range": "stddev: 0.00004723615080191593",
            "extra": "mean: 423.64922889181156 usec\nrounds: 1516"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2410.5062916583433,
            "unit": "iter/sec",
            "range": "stddev: 0.00004322724088649245",
            "extra": "mean: 414.85060771695197 usec\nrounds: 1555"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2112.830474708036,
            "unit": "iter/sec",
            "range": "stddev: 0.002601719433125255",
            "extra": "mean: 473.29873928393914 usec\nrounds: 1983"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2447.799611706031,
            "unit": "iter/sec",
            "range": "stddev: 0.00004687403392893968",
            "extra": "mean: 408.5301734740593 usec\nrounds: 2179"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2374.103218494104,
            "unit": "iter/sec",
            "range": "stddev: 0.00004331401607390398",
            "extra": "mean: 421.2116778285238 usec\nrounds: 1971"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2360.3979157095678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000467498564375252",
            "extra": "mean: 423.6573813866407 usec\nrounds: 2192"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2067.8530200768923,
            "unit": "iter/sec",
            "range": "stddev: 0.0028907027133328374",
            "extra": "mean: 483.5933648527956 usec\nrounds: 2242"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2420.6022990651577,
            "unit": "iter/sec",
            "range": "stddev: 0.000046093364643652466",
            "extra": "mean: 413.12032149444883 usec\nrounds: 2168"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2396.4725388069505,
            "unit": "iter/sec",
            "range": "stddev: 0.00005027191751237988",
            "extra": "mean: 417.2799745486905 usec\nrounds: 2161"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2368.718761013242,
            "unit": "iter/sec",
            "range": "stddev: 0.00004604190732265922",
            "extra": "mean: 422.16915594160304 usec\nrounds: 2129"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2355.927701575497,
            "unit": "iter/sec",
            "range": "stddev: 0.000060646505358731806",
            "extra": "mean: 424.4612427330697 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 1887.379725792224,
            "unit": "iter/sec",
            "range": "stddev: 0.003942095230421582",
            "extra": "mean: 529.8350863551065 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2373.9575140186043,
            "unit": "iter/sec",
            "range": "stddev: 0.00004580857657280086",
            "extra": "mean: 421.2375301979238 usec\nrounds: 2020"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2384.225523412341,
            "unit": "iter/sec",
            "range": "stddev: 0.00003983682825361853",
            "extra": "mean: 419.4234103193327 usec\nrounds: 1628"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2379.539492656718,
            "unit": "iter/sec",
            "range": "stddev: 0.00004405771975172559",
            "extra": "mean: 420.24938148158907 usec\nrounds: 1890"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2467.0548976716505,
            "unit": "iter/sec",
            "range": "stddev: 0.00004152603876104085",
            "extra": "mean: 405.341608305424 usec\nrounds: 2119"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2472.964231854464,
            "unit": "iter/sec",
            "range": "stddev: 0.000041360226164036714",
            "extra": "mean: 404.37301402054845 usec\nrounds: 2211"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 1891.692655779509,
            "unit": "iter/sec",
            "range": "stddev: 0.004760288114672587",
            "extra": "mean: 528.6270985642382 usec\nrounds: 2019"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2375.174513089971,
            "unit": "iter/sec",
            "range": "stddev: 0.00004841088126894844",
            "extra": "mean: 421.02169524337614 usec\nrounds: 2018"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2392.3301314513315,
            "unit": "iter/sec",
            "range": "stddev: 0.000044740205077624914",
            "extra": "mean: 418.002510127371 usec\nrounds: 2123"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2403.286020127879,
            "unit": "iter/sec",
            "range": "stddev: 0.000046174722680185716",
            "extra": "mean: 416.0969570932677 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2404.672199069623,
            "unit": "iter/sec",
            "range": "stddev: 0.000042521068623452456",
            "extra": "mean: 415.8570970242447 usec\nrounds: 1546"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2342.5399870757283,
            "unit": "iter/sec",
            "range": "stddev: 0.0000746346528223532",
            "extra": "mean: 426.8870565784167 usec\nrounds: 2227"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2489.663796016509,
            "unit": "iter/sec",
            "range": "stddev: 0.00003847158195000239",
            "extra": "mean: 401.66065859977226 usec\nrounds: 1157"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.462106150186793,
            "unit": "iter/sec",
            "range": "stddev: 0.006534893387265886",
            "extra": "mean: 406.1563308000075 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.16428765092474,
            "unit": "iter/sec",
            "range": "stddev: 0.000730306636806307",
            "extra": "mean: 47.249405058823534 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.15151657658365125,
            "unit": "iter/sec",
            "range": "stddev: 0.5093669057262457",
            "extra": "mean: 6.599937924599999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06963697143227415,
            "unit": "iter/sec",
            "range": "stddev: 0.09894641730830403",
            "extra": "mean: 14.360187978199997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.322774028947418,
            "unit": "iter/sec",
            "range": "stddev: 0.006541591588080539",
            "extra": "mean: 231.33293420000882 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 16.964168605388473,
            "unit": "iter/sec",
            "range": "stddev: 0.0030836102452555985",
            "extra": "mean: 58.9477753529496 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.44870159696752515,
            "unit": "iter/sec",
            "range": "stddev: 0.03004128853188743",
            "extra": "mean: 2.2286526430000095 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 41.86784331901563,
            "unit": "iter/sec",
            "range": "stddev: 0.0007630488864646336",
            "extra": "mean: 23.884679045452952 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 94.32622029813862,
            "unit": "iter/sec",
            "range": "stddev: 0.0005303211499610309",
            "extra": "mean: 10.601506101265178 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.847130846360653,
            "unit": "iter/sec",
            "range": "stddev: 0.004624066852924094",
            "extra": "mean: 101.55242329998941 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.42156351880025,
            "unit": "iter/sec",
            "range": "stddev: 0.0005178964566943981",
            "extra": "mean: 18.043518380003434 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 23.522456202072146,
            "unit": "iter/sec",
            "range": "stddev: 0.0005000503425647791",
            "extra": "mean: 42.51256720001493 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.32105351987590747,
            "unit": "iter/sec",
            "range": "stddev: 0.034482594591569",
            "extra": "mean: 3.1147454804000176 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.23124650693826462,
            "unit": "iter/sec",
            "range": "stddev: 1.0612215456368863",
            "extra": "mean: 4.324389644799988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5814410187305039,
            "unit": "iter/sec",
            "range": "stddev: 0.015373734858701334",
            "extra": "mean: 1.7198649008000189 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.098356514615404,
            "unit": "iter/sec",
            "range": "stddev: 0.0009202828398073209",
            "extra": "mean: 47.397056699998075 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 42.86569593229492,
            "unit": "iter/sec",
            "range": "stddev: 0.0009094980010605619",
            "extra": "mean: 23.32867758823909 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5292.389238961645,
            "unit": "iter/sec",
            "range": "stddev: 0.000017908641929345406",
            "extra": "mean: 188.9505769224559 usec\nrounds: 286"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9726239796326888,
            "unit": "iter/sec",
            "range": "stddev: 0.006613672748637377",
            "extra": "mean: 1.028146561199992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 22.713849100568105,
            "unit": "iter/sec",
            "range": "stddev: 0.0017385588804324707",
            "extra": "mean: 44.02600349999633 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 105.3526078036001,
            "unit": "iter/sec",
            "range": "stddev: 0.0003866461755965258",
            "extra": "mean: 9.491934000003255 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 19.855603952390187,
            "unit": "iter/sec",
            "range": "stddev: 0.006339233487926471",
            "extra": "mean: 50.36361534999401 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 34.3122697308485,
            "unit": "iter/sec",
            "range": "stddev: 0.0006480137819402065",
            "extra": "mean: 29.144093580639712 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.4030332679065955,
            "unit": "iter/sec",
            "range": "stddev: 0.010309179622233954",
            "extra": "mean: 712.7414744000021 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 3.972299629321851,
            "unit": "iter/sec",
            "range": "stddev: 0.00641615286262337",
            "extra": "mean: 251.74334599998926 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.759531057871132,
            "unit": "iter/sec",
            "range": "stddev: 0.025411874857110633",
            "extra": "mean: 265.99062079999385 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.773641103177274,
            "unit": "iter/sec",
            "range": "stddev: 0.005701408568773475",
            "extra": "mean: 360.5369126000028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.287651467473151,
            "unit": "iter/sec",
            "range": "stddev: 0.019406612218419113",
            "extra": "mean: 233.22791219999317 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 22.038016192897867,
            "unit": "iter/sec",
            "range": "stddev: 0.0010287845213322415",
            "extra": "mean: 45.37613509523908 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5442015070385159,
            "unit": "iter/sec",
            "range": "stddev: 0.03924392674524876",
            "extra": "mean: 1.8375546319999898 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 67.68882060327797,
            "unit": "iter/sec",
            "range": "stddev: 0.0005913078642820925",
            "extra": "mean: 14.773488311474182 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 94.2403151806532,
            "unit": "iter/sec",
            "range": "stddev: 0.0004798819355457121",
            "extra": "mean: 10.611169944445306 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.26163700822695063,
            "unit": "iter/sec",
            "range": "stddev: 1.183997871474901",
            "extra": "mean: 3.822089263200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 210.1813019635425,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348485925906242",
            "extra": "mean: 4.757797152543366 msec\nrounds: 118"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 10.945626000675093,
            "unit": "iter/sec",
            "range": "stddev: 0.001424616535625709",
            "extra": "mean: 91.36069512500455 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.844175113690996,
            "unit": "iter/sec",
            "range": "stddev: 0.0012292607668033604",
            "extra": "mean: 92.21540500000587 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.637373141008979,
            "unit": "iter/sec",
            "range": "stddev: 0.0015996006047768534",
            "extra": "mean: 94.00817163636206 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.476299309974975,
            "unit": "iter/sec",
            "range": "stddev: 0.005233372194670318",
            "extra": "mean: 95.45355381817444 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.700022731827048,
            "unit": "iter/sec",
            "range": "stddev: 0.017857381621993598",
            "extra": "mean: 103.09254190908942 msec\nrounds: 11"
          }
        ]
      }
    ]
  }
}