window.BENCHMARK_DATA = {
  "lastUpdate": 1773835825672,
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
      },
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
          "id": "9cc9a01ba08fde69dfdd547265aa39d613fb18f2",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/9cc9a01ba08fde69dfdd547265aa39d613fb18f2"
        },
        "date": 1773416401989,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4532.399649709938,
            "unit": "iter/sec",
            "range": "stddev: 0.0005671817946736956",
            "extra": "mean: 220.63367692299545 usec\nrounds: 1040"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4159.035352572387,
            "unit": "iter/sec",
            "range": "stddev: 0.0005356206827987529",
            "extra": "mean: 240.44037023669307 usec\nrounds: 2493"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2007.8760464849051,
            "unit": "iter/sec",
            "range": "stddev: 0.00014155648158246147",
            "extra": "mean: 498.038711976595 usec\nrounds: 1361"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2041.5848963282997,
            "unit": "iter/sec",
            "range": "stddev: 0.0009429986461922477",
            "extra": "mean: 489.8155358606227 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2207.959951283258,
            "unit": "iter/sec",
            "range": "stddev: 0.00005306797293469387",
            "extra": "mean: 452.9067655501649 usec\nrounds: 1881"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2039.9692507023888,
            "unit": "iter/sec",
            "range": "stddev: 0.0013894909723943841",
            "extra": "mean: 490.2034673589745 usec\nrounds: 1348"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2330.713062646882,
            "unit": "iter/sec",
            "range": "stddev: 0.00005028652397137891",
            "extra": "mean: 429.05324384475995 usec\nrounds: 2112"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2045.268718362749,
            "unit": "iter/sec",
            "range": "stddev: 0.0016784057531761675",
            "extra": "mean: 488.93330789340314 usec\nrounds: 1913"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2223.063537021876,
            "unit": "iter/sec",
            "range": "stddev: 0.00005410113299846555",
            "extra": "mean: 449.8296982279006 usec\nrounds: 1975"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2009.2048990183425,
            "unit": "iter/sec",
            "range": "stddev: 0.0021226368249085345",
            "extra": "mean: 497.7093179936899 usec\nrounds: 1934"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2259.386663370705,
            "unit": "iter/sec",
            "range": "stddev: 0.000048180103835735606",
            "extra": "mean: 442.59799184090645 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2246.5498643663664,
            "unit": "iter/sec",
            "range": "stddev: 0.00005295691948102846",
            "extra": "mean: 445.12699934307824 usec\nrounds: 1522"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2004.2847568755055,
            "unit": "iter/sec",
            "range": "stddev: 0.0023720252653093576",
            "extra": "mean: 498.9311007677909 usec\nrounds: 2084"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2296.746882126725,
            "unit": "iter/sec",
            "range": "stddev: 0.000048400929403286195",
            "extra": "mean: 435.3984358407085 usec\nrounds: 1356"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2281.5732244876435,
            "unit": "iter/sec",
            "range": "stddev: 0.00004920852449579957",
            "extra": "mean: 438.2940636168111 usec\nrounds: 1399"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2286.869459191026,
            "unit": "iter/sec",
            "range": "stddev: 0.00006603956359003455",
            "extra": "mean: 437.2790042653975 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 1948.467671513018,
            "unit": "iter/sec",
            "range": "stddev: 0.00292325776581055",
            "extra": "mean: 513.22380895521 usec\nrounds: 2010"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2233.402335729175,
            "unit": "iter/sec",
            "range": "stddev: 0.00005768101449254371",
            "extra": "mean: 447.74736016093306 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2235.3091211664596,
            "unit": "iter/sec",
            "range": "stddev: 0.00004992422782386353",
            "extra": "mean: 447.3654182908565 usec\nrounds: 2001"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2288.6333502077114,
            "unit": "iter/sec",
            "range": "stddev: 0.00005176686737877135",
            "extra": "mean: 436.9419854470102 usec\nrounds: 1924"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 1675.2407961469855,
            "unit": "iter/sec",
            "range": "stddev: 0.005091901302477983",
            "extra": "mean: 596.9291115044335 usec\nrounds: 1130"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2237.7222881177,
            "unit": "iter/sec",
            "range": "stddev: 0.000050805282120835894",
            "extra": "mean: 446.8829779772037 usec\nrounds: 1226"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2239.602808735054,
            "unit": "iter/sec",
            "range": "stddev: 0.000051195055093162624",
            "extra": "mean: 446.50774507860535 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2240.058830504545,
            "unit": "iter/sec",
            "range": "stddev: 0.00004991104811461148",
            "extra": "mean: 446.4168469069907 usec\nrounds: 1940"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2236.6977065688243,
            "unit": "iter/sec",
            "range": "stddev: 0.000057956246347067235",
            "extra": "mean: 447.0876851454533 usec\nrounds: 1858"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2254.155304921733,
            "unit": "iter/sec",
            "range": "stddev: 0.000046207381201294285",
            "extra": "mean: 443.6251565349536 usec\nrounds: 1974"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 1777.5681839960778,
            "unit": "iter/sec",
            "range": "stddev: 0.004950143358658022",
            "extra": "mean: 562.5663246019295 usec\nrounds: 1947"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2343.5959151054303,
            "unit": "iter/sec",
            "range": "stddev: 0.00004954352845220896",
            "extra": "mean: 426.69471880992484 usec\nrounds: 2084"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2340.6746742156215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475604975282927",
            "extra": "mean: 427.22724820146476 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2208.6314701186952,
            "unit": "iter/sec",
            "range": "stddev: 0.000064510376995157",
            "extra": "mean: 452.7690624394927 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2232.584244377278,
            "unit": "iter/sec",
            "range": "stddev: 0.000053249506351692785",
            "extra": "mean: 447.9114293305981 usec\nrounds: 1882"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2243.453400321345,
            "unit": "iter/sec",
            "range": "stddev: 0.00004861098988902597",
            "extra": "mean: 445.7413734810643 usec\nrounds: 1893"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 1718.7021738238825,
            "unit": "iter/sec",
            "range": "stddev: 0.006161236668380143",
            "extra": "mean: 581.834372022195 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2108.522808746086,
            "unit": "iter/sec",
            "range": "stddev: 0.00009558156509856917",
            "extra": "mean: 474.26567825210697 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2315.240671543233,
            "unit": "iter/sec",
            "range": "stddev: 0.00004775067589571687",
            "extra": "mean: 431.9205395322664 usec\nrounds: 2011"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2316.859541411604,
            "unit": "iter/sec",
            "range": "stddev: 0.000048784426601051745",
            "extra": "mean: 431.6187417173876 usec\nrounds: 1479"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.3723033155204845,
            "unit": "iter/sec",
            "range": "stddev: 0.006294681792734656",
            "extra": "mean: 421.53125760000023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 20.971911840532687,
            "unit": "iter/sec",
            "range": "stddev: 0.0011081942346158427",
            "extra": "mean: 47.68282489473787 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.14239785004873473,
            "unit": "iter/sec",
            "range": "stddev: 0.670231915605615",
            "extra": "mean: 7.022577936800005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.0649113496897904,
            "unit": "iter/sec",
            "range": "stddev: 0.2678786077183531",
            "extra": "mean: 15.4056263624 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.039219786858872,
            "unit": "iter/sec",
            "range": "stddev: 0.0034562680314823998",
            "extra": "mean: 247.57256419999294 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 16.6751870966913,
            "unit": "iter/sec",
            "range": "stddev: 0.0010785638409350584",
            "extra": "mean: 59.96934212500804 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.46004005934252545,
            "unit": "iter/sec",
            "range": "stddev: 0.011491817598562982",
            "extra": "mean: 2.1737237435999988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 43.03401681595684,
            "unit": "iter/sec",
            "range": "stddev: 0.0005590766225713848",
            "extra": "mean: 23.237431083337867 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 95.48950649038903,
            "unit": "iter/sec",
            "range": "stddev: 0.0009502683720136845",
            "extra": "mean: 10.47235488750431 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.73011843902001,
            "unit": "iter/sec",
            "range": "stddev: 0.000998356866839247",
            "extra": "mean: 102.77367190000177 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.14307502258204,
            "unit": "iter/sec",
            "range": "stddev: 0.0006511958864450148",
            "extra": "mean: 18.134643372544655 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 22.795404564981386,
            "unit": "iter/sec",
            "range": "stddev: 0.0004430517581132925",
            "extra": "mean: 43.86849099998926 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.31177875517115433,
            "unit": "iter/sec",
            "range": "stddev: 0.04358905483947036",
            "extra": "mean: 3.207402632200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.24628765276544584,
            "unit": "iter/sec",
            "range": "stddev: 0.9300913564586589",
            "extra": "mean: 4.0602928679999994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5688889540041745,
            "unit": "iter/sec",
            "range": "stddev: 0.014030545290267642",
            "extra": "mean: 1.7578122987999905 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.063622665946394,
            "unit": "iter/sec",
            "range": "stddev: 0.0012097465229560292",
            "extra": "mean: 47.47521429999324 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 43.653228086076304,
            "unit": "iter/sec",
            "range": "stddev: 0.002284557626362638",
            "extra": "mean: 22.907813324324607 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5327.024275961955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000744066309601039",
            "extra": "mean: 187.7220654901971 usec\nrounds: 397"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9440927910798882,
            "unit": "iter/sec",
            "range": "stddev: 0.005908148082550264",
            "extra": "mean: 1.059217917400008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 22.639401771671405,
            "unit": "iter/sec",
            "range": "stddev: 0.0007951829366907348",
            "extra": "mean: 44.17077845454804 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 101.95984509175418,
            "unit": "iter/sec",
            "range": "stddev: 0.0004940780350725486",
            "extra": "mean: 9.807782653064006 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 20.75056454064004,
            "unit": "iter/sec",
            "range": "stddev: 0.0009335824829585214",
            "extra": "mean: 48.19145994999303 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 31.345068711823977,
            "unit": "iter/sec",
            "range": "stddev: 0.0011614684728669187",
            "extra": "mean: 31.902944899998904 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.3227180857852112,
            "unit": "iter/sec",
            "range": "stddev: 0.015425943739124478",
            "extra": "mean: 756.0189965999939 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 3.8048328752803693,
            "unit": "iter/sec",
            "range": "stddev: 0.013731649129828183",
            "extra": "mean: 262.82363320000286 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.5403196561699475,
            "unit": "iter/sec",
            "range": "stddev: 0.023440885144128246",
            "extra": "mean: 282.4603699999898 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.594585986291222,
            "unit": "iter/sec",
            "range": "stddev: 0.004248965609395045",
            "extra": "mean: 385.4179454000018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.2215175192367855,
            "unit": "iter/sec",
            "range": "stddev: 0.003366631450638571",
            "extra": "mean: 236.88164160000724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 21.433032714330217,
            "unit": "iter/sec",
            "range": "stddev: 0.001137827351916596",
            "extra": "mean: 46.65695300000152 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5277781489465629,
            "unit": "iter/sec",
            "range": "stddev: 0.017041748859566365",
            "extra": "mean: 1.8947355095999796 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 68.47958086187,
            "unit": "iter/sec",
            "range": "stddev: 0.000856847851093623",
            "extra": "mean: 14.602893116666374 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 95.92878419737593,
            "unit": "iter/sec",
            "range": "stddev: 0.00041032505553654",
            "extra": "mean: 10.424399812495011 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.2375186829143858,
            "unit": "iter/sec",
            "range": "stddev: 1.8872251385238603",
            "extra": "mean: 4.210195121200013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 196.9527641100311,
            "unit": "iter/sec",
            "range": "stddev: 0.0005320929104928304",
            "extra": "mean: 5.077359561409011 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 9.360536158032813,
            "unit": "iter/sec",
            "range": "stddev: 0.0015437963375775664",
            "extra": "mean: 106.8314873333236 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 9.457831815170465,
            "unit": "iter/sec",
            "range": "stddev: 0.0028384090696202662",
            "extra": "mean: 105.73247860000947 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 8.982918599407043,
            "unit": "iter/sec",
            "range": "stddev: 0.018080520375672853",
            "extra": "mean: 111.3223936000054 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 9.58795894105343,
            "unit": "iter/sec",
            "range": "stddev: 0.002035887793031041",
            "extra": "mean: 104.29748460000496 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.483040523688802,
            "unit": "iter/sec",
            "range": "stddev: 0.0019614072662868373",
            "extra": "mean: 105.4514106000056 msec\nrounds: 10"
          }
        ]
      },
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
          "id": "915571fb33939ba6d4889fa48ba71cb3cfa9b777",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/915571fb33939ba6d4889fa48ba71cb3cfa9b777"
        },
        "date": 1773417249689,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4905.953302601473,
            "unit": "iter/sec",
            "range": "stddev: 0.00024428594550312285",
            "extra": "mean: 203.83398257576795 usec\nrounds: 1320"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4368.484784810229,
            "unit": "iter/sec",
            "range": "stddev: 0.00023315368232880852",
            "extra": "mean: 228.91232298144334 usec\nrounds: 2737"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2241.077223069803,
            "unit": "iter/sec",
            "range": "stddev: 0.000052942936616925615",
            "extra": "mean: 446.2139857145177 usec\nrounds: 1400"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2122.510652399688,
            "unit": "iter/sec",
            "range": "stddev: 0.00048539448920603223",
            "extra": "mean: 471.1401560550053 usec\nrounds: 1602"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2026.9091183382989,
            "unit": "iter/sec",
            "range": "stddev: 0.0005404576475715384",
            "extra": "mean: 493.3620313572916 usec\nrounds: 2041"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2148.5003795591592,
            "unit": "iter/sec",
            "range": "stddev: 0.00007211871727709228",
            "extra": "mean: 465.44092312666254 usec\nrounds: 1548"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2279.095486731657,
            "unit": "iter/sec",
            "range": "stddev: 0.0006699743327382247",
            "extra": "mean: 438.77055868073904 usec\nrounds: 2062"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2268.8365535673042,
            "unit": "iter/sec",
            "range": "stddev: 0.00004869821776686472",
            "extra": "mean: 440.754534930114 usec\nrounds: 2004"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2189.7568276347138,
            "unit": "iter/sec",
            "range": "stddev: 0.000922579889844541",
            "extra": "mean: 456.67171230157066 usec\nrounds: 2016"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2283.737189877794,
            "unit": "iter/sec",
            "range": "stddev: 0.00004470927410577139",
            "extra": "mean: 437.87875611620234 usec\nrounds: 1308"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2284.086001057284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377657819850504",
            "extra": "mean: 437.8118860398025 usec\nrounds: 2106"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2279.017652759296,
            "unit": "iter/sec",
            "range": "stddev: 0.000036591263930011414",
            "extra": "mean: 438.7855437579699 usec\nrounds: 1554"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2138.33569966467,
            "unit": "iter/sec",
            "range": "stddev: 0.0011398971081352824",
            "extra": "mean: 467.6534185707222 usec\nrounds: 2057"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2316.2947911400706,
            "unit": "iter/sec",
            "range": "stddev: 0.000036778045347789244",
            "extra": "mean: 431.72397737327907 usec\nrounds: 2033"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2256.6466369768327,
            "unit": "iter/sec",
            "range": "stddev: 0.00005949873430636982",
            "extra": "mean: 443.1353955086528 usec\nrounds: 1603"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2144.627146558281,
            "unit": "iter/sec",
            "range": "stddev: 0.0017526966327176775",
            "extra": "mean: 466.28151732799336 usec\nrounds: 1991"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2249.2283017378663,
            "unit": "iter/sec",
            "range": "stddev: 0.000042579377532018526",
            "extra": "mean: 444.5969309684348 usec\nrounds: 2086"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2248.9508061984566,
            "unit": "iter/sec",
            "range": "stddev: 0.00004934867948669874",
            "extra": "mean: 444.6517892894078 usec\nrounds: 1998"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2267.07236290545,
            "unit": "iter/sec",
            "range": "stddev: 0.000038404755681580605",
            "extra": "mean: 441.09752135058153 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2032.491135527727,
            "unit": "iter/sec",
            "range": "stddev: 0.002533406460927954",
            "extra": "mean: 492.00706587109147 usec\nrounds: 2095"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2300.848753499685,
            "unit": "iter/sec",
            "range": "stddev: 0.000036993234406836596",
            "extra": "mean: 434.62222298573914 usec\nrounds: 1601"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2291.13396638293,
            "unit": "iter/sec",
            "range": "stddev: 0.00003950940636637995",
            "extra": "mean: 436.46509312535954 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2263.0603912601982,
            "unit": "iter/sec",
            "range": "stddev: 0.00003947871456895811",
            "extra": "mean: 441.8795025806378 usec\nrounds: 1550"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2254.636576755338,
            "unit": "iter/sec",
            "range": "stddev: 0.000046477651167578425",
            "extra": "mean: 443.53046087769343 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 1901.2065265039826,
            "unit": "iter/sec",
            "range": "stddev: 0.003831358987270397",
            "extra": "mean: 525.981783703868 usec\nrounds: 2025"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2250.8448976637105,
            "unit": "iter/sec",
            "range": "stddev: 0.00004406555113308477",
            "extra": "mean: 444.2776137253887 usec\nrounds: 2040"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2271.6836472034934,
            "unit": "iter/sec",
            "range": "stddev: 0.000038282439190384395",
            "extra": "mean: 440.202138722541 usec\nrounds: 2004"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2363.2232820561626,
            "unit": "iter/sec",
            "range": "stddev: 0.000037469266709242",
            "extra": "mean: 423.1508751597661 usec\nrounds: 1562"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2361.8912409551904,
            "unit": "iter/sec",
            "range": "stddev: 0.00003754223692340834",
            "extra": "mean: 423.38952050797326 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2271.656624807693,
            "unit": "iter/sec",
            "range": "stddev: 0.00003932482960290947",
            "extra": "mean: 440.2073751285606 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2239.4671845828193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441455380846429",
            "extra": "mean: 446.5347859903049 usec\nrounds: 2070"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 1828.4541769759892,
            "unit": "iter/sec",
            "range": "stddev: 0.00451963485786658",
            "extra": "mean: 546.9100689489862 usec\nrounds: 2074"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2312.157728186853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000374754548774744",
            "extra": "mean: 432.49644598605283 usec\nrounds: 2018"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2303.3246658708395,
            "unit": "iter/sec",
            "range": "stddev: 0.00003649039594509959",
            "extra": "mean: 434.1550345973136 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2347.264444472613,
            "unit": "iter/sec",
            "range": "stddev: 0.00003726390257385394",
            "extra": "mean: 426.0278394941059 usec\nrounds: 2056"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2274.0284804893645,
            "unit": "iter/sec",
            "range": "stddev: 0.00006365431893911494",
            "extra": "mean: 439.74823032330835 usec\nrounds: 2071"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.5337262848889095,
            "unit": "iter/sec",
            "range": "stddev: 0.004193107562912332",
            "extra": "mean: 394.6756229999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.938410466246204,
            "unit": "iter/sec",
            "range": "stddev: 0.0008852981937249496",
            "extra": "mean: 45.582153800001635 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1525209396250435,
            "unit": "iter/sec",
            "range": "stddev: 0.523729940519325",
            "extra": "mean: 6.556476785799992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06956783786826565,
            "unit": "iter/sec",
            "range": "stddev: 0.041957504628880024",
            "extra": "mean: 14.374458523399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.343892952521648,
            "unit": "iter/sec",
            "range": "stddev: 0.00669476034553635",
            "extra": "mean: 230.208251199997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 18.951580858750656,
            "unit": "iter/sec",
            "range": "stddev: 0.0011122717774288762",
            "extra": "mean: 52.76604666666963 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5086623190327714,
            "unit": "iter/sec",
            "range": "stddev: 0.013605048192735524",
            "extra": "mean: 1.9659407874000068 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 45.12062550730604,
            "unit": "iter/sec",
            "range": "stddev: 0.00038868859141540915",
            "extra": "mean: 22.162813319998804 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 104.3951969961895,
            "unit": "iter/sec",
            "range": "stddev: 0.00024093154625639957",
            "extra": "mean: 9.578984750002443 msec\nrounds: 84"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 10.466650005092822,
            "unit": "iter/sec",
            "range": "stddev: 0.0009255135260007284",
            "extra": "mean: 95.54155336362865 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 58.218668204913435,
            "unit": "iter/sec",
            "range": "stddev: 0.0005323056052123113",
            "extra": "mean: 17.176621019228396 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 25.647701014201505,
            "unit": "iter/sec",
            "range": "stddev: 0.0006197335297576265",
            "extra": "mean: 38.989849400002186 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.337292853058988,
            "unit": "iter/sec",
            "range": "stddev: 0.0052029846465765",
            "extra": "mean: 2.9647826538000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.23554153177998918,
            "unit": "iter/sec",
            "range": "stddev: 1.0122152651448064",
            "extra": "mean: 4.245535776399993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5970847460036979,
            "unit": "iter/sec",
            "range": "stddev: 0.022617503803188808",
            "extra": "mean: 1.6748041324000043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.93271075703469,
            "unit": "iter/sec",
            "range": "stddev: 0.0008348304816662326",
            "extra": "mean: 45.59399935000101 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 45.047148056946895,
            "unit": "iter/sec",
            "range": "stddev: 0.0006444387320512205",
            "extra": "mean: 22.198963600000557 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5257.234922723434,
            "unit": "iter/sec",
            "range": "stddev: 0.00006499390243962863",
            "extra": "mean: 190.21406018545667 usec\nrounds: 432"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9672461934163996,
            "unit": "iter/sec",
            "range": "stddev: 0.007194203367956635",
            "extra": "mean: 1.0338629469999887 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 23.142603127839994,
            "unit": "iter/sec",
            "range": "stddev: 0.006285209517353488",
            "extra": "mean: 43.21035081818536 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 111.32716048733235,
            "unit": "iter/sec",
            "range": "stddev: 0.0003015124986552408",
            "extra": "mean: 8.982533962265098 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 21.24910186117509,
            "unit": "iter/sec",
            "range": "stddev: 0.0019626322994678353",
            "extra": "mean: 47.06081257143069 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 36.325656418806226,
            "unit": "iter/sec",
            "range": "stddev: 0.0005748725157420852",
            "extra": "mean: 27.528752363640375 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.487650767893546,
            "unit": "iter/sec",
            "range": "stddev: 0.0024302588094929846",
            "extra": "mean: 672.2007756000153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.411126784295173,
            "unit": "iter/sec",
            "range": "stddev: 0.002729868551617976",
            "extra": "mean: 226.6994464000163 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 1.3141270355487866,
            "unit": "iter/sec",
            "range": "stddev: 1.1448594154142853",
            "extra": "mean: 760.9614389999933 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.868610251201671,
            "unit": "iter/sec",
            "range": "stddev: 0.006397934461274562",
            "extra": "mean: 348.6008598000012 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.734612910954813,
            "unit": "iter/sec",
            "range": "stddev: 0.004179969716531171",
            "extra": "mean: 211.21050840000635 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 23.97054850012529,
            "unit": "iter/sec",
            "range": "stddev: 0.0006968635623338259",
            "extra": "mean: 41.71786056521707 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5600071783064154,
            "unit": "iter/sec",
            "range": "stddev: 0.028533098767436933",
            "extra": "mean: 1.785691395999993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 72.07438306038057,
            "unit": "iter/sec",
            "range": "stddev: 0.0003636182653981545",
            "extra": "mean: 13.874555112906709 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 100.80504346025863,
            "unit": "iter/sec",
            "range": "stddev: 0.0002589467789714377",
            "extra": "mean: 9.920138573168115 msec\nrounds: 82"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.30213071992646995,
            "unit": "iter/sec",
            "range": "stddev: 0.2311976516668723",
            "extra": "mean: 3.3098256286000036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 211.17345978180495,
            "unit": "iter/sec",
            "range": "stddev: 0.000970471753847443",
            "extra": "mean: 4.735443559210757 msec\nrounds: 152"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 9.952996399441219,
            "unit": "iter/sec",
            "range": "stddev: 0.012890167849211667",
            "extra": "mean: 100.47225577778187 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.530707850439244,
            "unit": "iter/sec",
            "range": "stddev: 0.001383709011782678",
            "extra": "mean: 94.96037818182272 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.366927278559293,
            "unit": "iter/sec",
            "range": "stddev: 0.0006242279648827203",
            "extra": "mean: 96.46059754544467 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.431269208796714,
            "unit": "iter/sec",
            "range": "stddev: 0.0010193374832716472",
            "extra": "mean: 95.865611363639 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 10.30791100177446,
            "unit": "iter/sec",
            "range": "stddev: 0.0005435676177378033",
            "extra": "mean: 97.01286709090274 msec\nrounds: 11"
          }
        ]
      },
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
          "id": "59f0bf26412f38d4b188abbf1ed4d5e94f11bfc6",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/59f0bf26412f38d4b188abbf1ed4d5e94f11bfc6"
        },
        "date": 1773423913898,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4884.911446605496,
            "unit": "iter/sec",
            "range": "stddev: 0.000306744890703723",
            "extra": "mean: 204.7120016259242 usec\nrounds: 1230"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4226.881729691844,
            "unit": "iter/sec",
            "range": "stddev: 0.0002822430114988803",
            "extra": "mean: 236.5810221221647 usec\nrounds: 2667"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2130.486492268651,
            "unit": "iter/sec",
            "range": "stddev: 0.00005471373470017291",
            "extra": "mean: 469.3763624547315 usec\nrounds: 1385"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2040.5826334586575,
            "unit": "iter/sec",
            "range": "stddev: 0.00042405895802993847",
            "extra": "mean: 490.0561161324125 usec\nrounds: 1903"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 1899.7319508581493,
            "unit": "iter/sec",
            "range": "stddev: 0.00011953693287239027",
            "extra": "mean: 526.390051790348 usec\nrounds: 1564"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 1973.0248110108846,
            "unit": "iter/sec",
            "range": "stddev: 0.0006001459286812571",
            "extra": "mean: 506.8359984219597 usec\nrounds: 1901"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2177.1674368647377,
            "unit": "iter/sec",
            "range": "stddev: 0.0006929088616704689",
            "extra": "mean: 459.312399711464 usec\nrounds: 2079"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2168.719186107401,
            "unit": "iter/sec",
            "range": "stddev: 0.00003445773111608702",
            "extra": "mean: 461.1016522590386 usec\nrounds: 1527"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2182.4488673161063,
            "unit": "iter/sec",
            "range": "stddev: 0.00003499017778632511",
            "extra": "mean: 458.20088386756225 usec\nrounds: 1903"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2033.2292684353617,
            "unit": "iter/sec",
            "range": "stddev: 0.0010328514048700802",
            "extra": "mean: 491.82845020204417 usec\nrounds: 1486"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2192.469737538527,
            "unit": "iter/sec",
            "range": "stddev: 0.00003655847102218597",
            "extra": "mean: 456.1066375870228 usec\nrounds: 2006"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2181.256152924709,
            "unit": "iter/sec",
            "range": "stddev: 0.000036946782447623066",
            "extra": "mean: 458.45142885174806 usec\nrounds: 2038"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 1965.3579183883476,
            "unit": "iter/sec",
            "range": "stddev: 0.0017140924638677803",
            "extra": "mean: 508.81317374497877 usec\nrounds: 1295"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2220.821623573385,
            "unit": "iter/sec",
            "range": "stddev: 0.000038986243598133936",
            "extra": "mean: 450.2838000969041 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2221.4280698808943,
            "unit": "iter/sec",
            "range": "stddev: 0.00003734820487058616",
            "extra": "mean: 450.16087334019187 usec\nrounds: 1958"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2080.882584414169,
            "unit": "iter/sec",
            "range": "stddev: 0.0016927240893945064",
            "extra": "mean: 480.56531756765617 usec\nrounds: 2072"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2181.8635458484364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000409784915780146",
            "extra": "mean: 458.3238039348338 usec\nrounds: 1474"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2179.7914768698333,
            "unit": "iter/sec",
            "range": "stddev: 0.00003702523460551733",
            "extra": "mean: 458.7594779643756 usec\nrounds: 1906"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2175.86620959662,
            "unit": "iter/sec",
            "range": "stddev: 0.000037782557239036224",
            "extra": "mean: 459.58708103904434 usec\nrounds: 1925"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2209.233909297327,
            "unit": "iter/sec",
            "range": "stddev: 0.00003990030964657258",
            "extra": "mean: 452.64559619133394 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 1954.0835499714544,
            "unit": "iter/sec",
            "range": "stddev: 0.0024822617776310966",
            "extra": "mean: 511.74884513746 usec\nrounds: 1892"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2174.8485120647283,
            "unit": "iter/sec",
            "range": "stddev: 0.000044795303241571916",
            "extra": "mean: 459.8021399893428 usec\nrounds: 1893"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2178.049947936487,
            "unit": "iter/sec",
            "range": "stddev: 0.000038094915678910434",
            "extra": "mean: 459.1262936588818 usec\nrounds: 1924"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2154.85509794128,
            "unit": "iter/sec",
            "range": "stddev: 0.00005785061757011197",
            "extra": "mean: 464.0683268937141 usec\nrounds: 1967"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2178.175204395634,
            "unit": "iter/sec",
            "range": "stddev: 0.000037450557070432625",
            "extra": "mean: 459.09989149723356 usec\nrounds: 1917"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 1850.5353132776706,
            "unit": "iter/sec",
            "range": "stddev: 0.0033787805630206102",
            "extra": "mean: 540.3841757706308 usec\nrounds: 1849"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2175.4730490844127,
            "unit": "iter/sec",
            "range": "stddev: 0.00003634807066457563",
            "extra": "mean: 459.6701395224676 usec\nrounds: 1885"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2257.3308674480654,
            "unit": "iter/sec",
            "range": "stddev: 0.00003747452596527389",
            "extra": "mean: 443.0010745967913 usec\nrounds: 1984"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2264.3414807744916,
            "unit": "iter/sec",
            "range": "stddev: 0.00003672959455037742",
            "extra": "mean: 441.6295017737173 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2185.1300585205854,
            "unit": "iter/sec",
            "range": "stddev: 0.00003680018871218009",
            "extra": "mean: 457.638663703632 usec\nrounds: 2025"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2161.434319098551,
            "unit": "iter/sec",
            "range": "stddev: 0.00004061621548725124",
            "extra": "mean: 462.6557426075573 usec\nrounds: 1488"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2174.408345611884,
            "unit": "iter/sec",
            "range": "stddev: 0.000039654094983380756",
            "extra": "mean: 459.89521794196264 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 1786.0955517123668,
            "unit": "iter/sec",
            "range": "stddev: 0.004654591370930166",
            "extra": "mean: 559.8804605057547 usec\nrounds: 1937"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2199.357919382077,
            "unit": "iter/sec",
            "range": "stddev: 0.00003813077100821147",
            "extra": "mean: 454.6781545592889 usec\nrounds: 1941"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2157.3147463863797,
            "unit": "iter/sec",
            "range": "stddev: 0.00006644305704438036",
            "extra": "mean: 463.53922239443955 usec\nrounds: 1938"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2243.001504872564,
            "unit": "iter/sec",
            "range": "stddev: 0.00004048957679642646",
            "extra": "mean: 445.83117658532956 usec\nrounds: 2050"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.5687572761141473,
            "unit": "iter/sec",
            "range": "stddev: 0.006140311483573305",
            "extra": "mean: 389.29330120000145 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 22.367327736415092,
            "unit": "iter/sec",
            "range": "stddev: 0.0008667993190854526",
            "extra": "mean: 44.70806757894246 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1545192514202771,
            "unit": "iter/sec",
            "range": "stddev: 0.5414798424854039",
            "extra": "mean: 6.471685507199998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06889649084881033,
            "unit": "iter/sec",
            "range": "stddev: 0.583060009506961",
            "extra": "mean: 14.514527339199997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.532269518307232,
            "unit": "iter/sec",
            "range": "stddev: 0.001992105650802557",
            "extra": "mean: 220.64001179998058 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 19.223703488505855,
            "unit": "iter/sec",
            "range": "stddev: 0.0004650605521978157",
            "extra": "mean: 52.01911278947448 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5016796727088361,
            "unit": "iter/sec",
            "range": "stddev: 0.041407110242573375",
            "extra": "mean: 1.9933038040000042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 43.931985270893264,
            "unit": "iter/sec",
            "range": "stddev: 0.001166193641205233",
            "extra": "mean: 22.762458692312748 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 100.55974029530009,
            "unit": "iter/sec",
            "range": "stddev: 0.0006785310334038877",
            "extra": "mean: 9.944337535711968 msec\nrounds: 84"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 10.30820009539654,
            "unit": "iter/sec",
            "range": "stddev: 0.0009704619181775536",
            "extra": "mean: 97.0101463636297 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.32118031288878,
            "unit": "iter/sec",
            "range": "stddev: 0.002469030215032552",
            "extra": "mean: 18.076259297870028 msec\nrounds: 47"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 25.47157721452268,
            "unit": "iter/sec",
            "range": "stddev: 0.0011724917994030632",
            "extra": "mean: 39.25944560000971 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.3321021232278606,
            "unit": "iter/sec",
            "range": "stddev: 0.024609538994772886",
            "extra": "mean: 3.0111219713999957 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.24294783570763542,
            "unit": "iter/sec",
            "range": "stddev: 0.9488370284202898",
            "extra": "mean: 4.116109934000008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5969164865389565,
            "unit": "iter/sec",
            "range": "stddev: 0.027093407237309852",
            "extra": "mean: 1.675276228000007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 23.120716508802783,
            "unit": "iter/sec",
            "range": "stddev: 0.0005314569999700133",
            "extra": "mean: 43.251254761904484 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 45.57259685115007,
            "unit": "iter/sec",
            "range": "stddev: 0.0007192014945824063",
            "extra": "mean: 21.94301113158453 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5008.978594176561,
            "unit": "iter/sec",
            "range": "stddev: 0.000027427681783534417",
            "extra": "mean: 199.64149999814336 usec\nrounds: 332"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9750141926965873,
            "unit": "iter/sec",
            "range": "stddev: 0.00493886129344474",
            "extra": "mean: 1.0256260960000076 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 21.82656578193677,
            "unit": "iter/sec",
            "range": "stddev: 0.005742990200305379",
            "extra": "mean: 45.81572795238268 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 103.69799641296021,
            "unit": "iter/sec",
            "range": "stddev: 0.00024184645769077746",
            "extra": "mean: 9.643387862747748 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 21.373966417636982,
            "unit": "iter/sec",
            "range": "stddev: 0.0022297052459431923",
            "extra": "mean: 46.7858880499989 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 36.15883399890475,
            "unit": "iter/sec",
            "range": "stddev: 0.000882629872627296",
            "extra": "mean: 27.655759033333048 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.4424313462605014,
            "unit": "iter/sec",
            "range": "stddev: 0.0019966645725118144",
            "extra": "mean: 693.2738965999988 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.414208002469335,
            "unit": "iter/sec",
            "range": "stddev: 0.0018591383667078579",
            "extra": "mean: 226.54120499998953 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 4.119239768695827,
            "unit": "iter/sec",
            "range": "stddev: 0.017241688826109405",
            "extra": "mean: 242.76324179998028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.928039188932599,
            "unit": "iter/sec",
            "range": "stddev: 0.0005348851018628157",
            "extra": "mean: 341.52548360001447 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.76897029115482,
            "unit": "iter/sec",
            "range": "stddev: 0.0009161320691695686",
            "extra": "mean: 209.68887180000593 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 23.657311589199473,
            "unit": "iter/sec",
            "range": "stddev: 0.001385730359160627",
            "extra": "mean: 42.27022991304476 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5438750682176238,
            "unit": "iter/sec",
            "range": "stddev: 0.03992777982764866",
            "extra": "mean: 1.838657549200002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 70.77070852834635,
            "unit": "iter/sec",
            "range": "stddev: 0.0006754736087674696",
            "extra": "mean: 14.130139725808485 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 100.97347530431499,
            "unit": "iter/sec",
            "range": "stddev: 0.0002555621461013376",
            "extra": "mean: 9.903590987496358 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.20475462090155802,
            "unit": "iter/sec",
            "range": "stddev: 3.747239777891612",
            "extra": "mean: 4.8838946617999905 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 217.58616221644826,
            "unit": "iter/sec",
            "range": "stddev: 0.000214183792204031",
            "extra": "mean: 4.59588049999811 msec\nrounds: 108"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 10.413043965572532,
            "unit": "iter/sec",
            "range": "stddev: 0.0002092126639952868",
            "extra": "mean: 96.03339842856583 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.302319341391588,
            "unit": "iter/sec",
            "range": "stddev: 0.0020468857536968333",
            "extra": "mean: 97.06552154545473 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.25711760062931,
            "unit": "iter/sec",
            "range": "stddev: 0.000946933345442246",
            "extra": "mean: 97.4932762727266 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.1836079329008,
            "unit": "iter/sec",
            "range": "stddev: 0.001397297381235884",
            "extra": "mean: 98.1970247272815 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.85015587383684,
            "unit": "iter/sec",
            "range": "stddev: 0.009268581262072929",
            "extra": "mean: 101.52123609090454 msec\nrounds: 11"
          }
        ]
      },
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
          "id": "51762794b1ae520871dd66dd0682510d4305d85e",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/51762794b1ae520871dd66dd0682510d4305d85e"
        },
        "date": 1773428581903,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4822.578528854923,
            "unit": "iter/sec",
            "range": "stddev: 0.0003325420653586069",
            "extra": "mean: 207.35795052723398 usec\nrounds: 1233"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4316.174901007702,
            "unit": "iter/sec",
            "range": "stddev: 0.00004048043894135907",
            "extra": "mean: 231.6866259906495 usec\nrounds: 1893"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2096.7270877699443,
            "unit": "iter/sec",
            "range": "stddev: 0.00038976011131209166",
            "extra": "mean: 476.93379163789456 usec\nrounds: 1459"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 1924.799138926662,
            "unit": "iter/sec",
            "range": "stddev: 0.0005144309865607468",
            "extra": "mean: 519.5347295082624 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2041.799232551979,
            "unit": "iter/sec",
            "range": "stddev: 0.00006834778080387428",
            "extra": "mean: 489.76411787075284 usec\nrounds: 1841"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2067.355860366854,
            "unit": "iter/sec",
            "range": "stddev: 0.0006017439568517839",
            "extra": "mean: 483.7096598466358 usec\nrounds: 1955"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2260.221155344394,
            "unit": "iter/sec",
            "range": "stddev: 0.000037735026245540304",
            "extra": "mean: 442.4345810742702 usec\nrounds: 1955"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2051.894781131948,
            "unit": "iter/sec",
            "range": "stddev: 0.0010051595133155461",
            "extra": "mean: 487.354424405885 usec\nrounds: 1852"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2178.9463750568234,
            "unit": "iter/sec",
            "range": "stddev: 0.00003799973340778374",
            "extra": "mean: 458.9374072934317 usec\nrounds: 1947"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2049.0041492625414,
            "unit": "iter/sec",
            "range": "stddev: 0.0011541088880507367",
            "extra": "mean: 488.04195948549483 usec\nrounds: 1555"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2184.987379952903,
            "unit": "iter/sec",
            "range": "stddev: 0.000037343209600880724",
            "extra": "mean: 457.6685472762569 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2170.622037976084,
            "unit": "iter/sec",
            "range": "stddev: 0.00003866606994360759",
            "extra": "mean: 460.6974325813134 usec\nrounds: 1995"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2024.2824672917195,
            "unit": "iter/sec",
            "range": "stddev: 0.0013616685222036559",
            "extra": "mean: 494.00220382182954 usec\nrounds: 1884"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2209.03793146568,
            "unit": "iter/sec",
            "range": "stddev: 0.00003805641267446871",
            "extra": "mean: 452.68575326658504 usec\nrounds: 1990"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2197.2457473394848,
            "unit": "iter/sec",
            "range": "stddev: 0.00004727254219256557",
            "extra": "mean: 455.1152283311236 usec\nrounds: 2019"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2031.7579955578708,
            "unit": "iter/sec",
            "range": "stddev: 0.002113184391744528",
            "extra": "mean: 492.184601801173 usec\nrounds: 1999"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2148.0593375078856,
            "unit": "iter/sec",
            "range": "stddev: 0.00005078309250606336",
            "extra": "mean: 465.53648800045266 usec\nrounds: 1375"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2165.8485602758733,
            "unit": "iter/sec",
            "range": "stddev: 0.00003831933343696477",
            "extra": "mean: 461.71279854978684 usec\nrounds: 1931"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2134.6594378026953,
            "unit": "iter/sec",
            "range": "stddev: 0.00004534305766980695",
            "extra": "mean: 468.4588006362957 usec\nrounds: 1886"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2193.451432908899,
            "unit": "iter/sec",
            "range": "stddev: 0.00004098826914408436",
            "extra": "mean: 455.90250369657184 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 1937.744798285123,
            "unit": "iter/sec",
            "range": "stddev: 0.002701338803817782",
            "extra": "mean: 516.0638288824132 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2172.252879529581,
            "unit": "iter/sec",
            "range": "stddev: 0.00004056070558761341",
            "extra": "mean: 460.3515591686351 usec\nrounds: 1876"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2167.3394893734626,
            "unit": "iter/sec",
            "range": "stddev: 0.00003960114518813538",
            "extra": "mean: 461.39518285115605 usec\nrounds: 1936"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2156.961730270655,
            "unit": "iter/sec",
            "range": "stddev: 0.000055243919783663173",
            "extra": "mean: 463.61508689100395 usec\nrounds: 1991"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2165.085688898979,
            "unit": "iter/sec",
            "range": "stddev: 0.00004063314377614623",
            "extra": "mean: 461.87548378675706 usec\nrounds: 1912"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 1865.5002777477666,
            "unit": "iter/sec",
            "range": "stddev: 0.0034065161052516772",
            "extra": "mean: 536.0492367266265 usec\nrounds: 2053"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2177.031575907815,
            "unit": "iter/sec",
            "range": "stddev: 0.00003676178583435242",
            "extra": "mean: 459.3410637983068 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2248.352855468972,
            "unit": "iter/sec",
            "range": "stddev: 0.00003975305740134756",
            "extra": "mean: 444.7700446874098 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2253.4261818247414,
            "unit": "iter/sec",
            "range": "stddev: 0.00003834319575754389",
            "extra": "mean: 443.7686967807559 usec\nrounds: 1926"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2076.7564941862292,
            "unit": "iter/sec",
            "range": "stddev: 0.00006931361544721545",
            "extra": "mean: 481.5201025249939 usec\nrounds: 1980"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2156.592931481081,
            "unit": "iter/sec",
            "range": "stddev: 0.00004000878611505361",
            "extra": "mean: 463.694369670975 usec\nrounds: 1853"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 1740.8478193920796,
            "unit": "iter/sec",
            "range": "stddev: 0.004439800738909434",
            "extra": "mean: 574.4327498708126 usec\nrounds: 1935"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2139.4197571812742,
            "unit": "iter/sec",
            "range": "stddev: 0.00006226972708061532",
            "extra": "mean: 467.4164556269775 usec\nrounds: 2017"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2140.7588097323123,
            "unit": "iter/sec",
            "range": "stddev: 0.00005950803071878158",
            "extra": "mean: 467.12408490568976 usec\nrounds: 1908"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2222.195388304858,
            "unit": "iter/sec",
            "range": "stddev: 0.000039514963374873807",
            "extra": "mean: 450.00543393388244 usec\nrounds: 1998"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2212.85957352773,
            "unit": "iter/sec",
            "range": "stddev: 0.00004535059075754207",
            "extra": "mean: 451.9039581015097 usec\nrounds: 1981"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.6106059233316543,
            "unit": "iter/sec",
            "range": "stddev: 0.00534679671802059",
            "extra": "mean: 383.05283500000655 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.713739942533763,
            "unit": "iter/sec",
            "range": "stddev: 0.0011373113786034379",
            "extra": "mean: 46.05378910526413 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1528291394903627,
            "unit": "iter/sec",
            "range": "stddev: 0.5037666844268821",
            "extra": "mean: 6.543254796399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06795189189703185,
            "unit": "iter/sec",
            "range": "stddev: 0.6264047976651606",
            "extra": "mean: 14.716293720200014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.269809118652657,
            "unit": "iter/sec",
            "range": "stddev: 0.012671813004793718",
            "extra": "mean: 234.20250700002043 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 19.1756046627972,
            "unit": "iter/sec",
            "range": "stddev: 0.0007442655996419052",
            "extra": "mean: 52.14959411111093 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5006077932783193,
            "unit": "iter/sec",
            "range": "stddev: 0.015082500136596348",
            "extra": "mean: 1.9975717786000131 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 44.52927933169205,
            "unit": "iter/sec",
            "range": "stddev: 0.0005121214686768503",
            "extra": "mean: 22.457134160001715 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 102.45433696913538,
            "unit": "iter/sec",
            "range": "stddev: 0.00039167847736760195",
            "extra": "mean: 9.760445771087781 msec\nrounds: 83"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 10.298679976055926,
            "unit": "iter/sec",
            "range": "stddev: 0.0033561619147533467",
            "extra": "mean: 97.09982272727818 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 58.089167211272475,
            "unit": "iter/sec",
            "range": "stddev: 0.00033222478048836946",
            "extra": "mean: 17.21491369230622 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 24.66080395838166,
            "unit": "iter/sec",
            "range": "stddev: 0.000798637142921336",
            "extra": "mean: 40.55017840000801 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.33586707913314184,
            "unit": "iter/sec",
            "range": "stddev: 0.012785080870451828",
            "extra": "mean: 2.9773683165999953 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.23598129531226963,
            "unit": "iter/sec",
            "range": "stddev: 1.0355756852311995",
            "extra": "mean: 4.237623997599973 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.6098132431599945,
            "unit": "iter/sec",
            "range": "stddev: 0.010701028581926427",
            "extra": "mean: 1.6398463155999934 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 23.01120986376194,
            "unit": "iter/sec",
            "range": "stddev: 0.0005335505001791892",
            "extra": "mean: 43.457080523818966 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 46.81688523949979,
            "unit": "iter/sec",
            "range": "stddev: 0.0019358543675955528",
            "extra": "mean: 21.359814837837433 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5386.748797799324,
            "unit": "iter/sec",
            "range": "stddev: 0.000014618089139948108",
            "extra": "mean: 185.64073387059278 usec\nrounds: 496"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9777374543555619,
            "unit": "iter/sec",
            "range": "stddev: 0.011272237435274581",
            "extra": "mean: 1.0227694515999814 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 24.258988545846893,
            "unit": "iter/sec",
            "range": "stddev: 0.0004633477108825355",
            "extra": "mean: 41.22183404761938 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 111.7347281752161,
            "unit": "iter/sec",
            "range": "stddev: 0.0002482845327890404",
            "extra": "mean: 8.949768942310007 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 21.674282543744887,
            "unit": "iter/sec",
            "range": "stddev: 0.0013618845646687302",
            "extra": "mean: 46.137628684212025 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 35.54860938121588,
            "unit": "iter/sec",
            "range": "stddev: 0.0017964541910977358",
            "extra": "mean: 28.130495606064592 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 0.8464483152611639,
            "unit": "iter/sec",
            "range": "stddev: 1.1382502600780613",
            "extra": "mean: 1.1814070416000049 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.349004988756636,
            "unit": "iter/sec",
            "range": "stddev: 0.004109534202164468",
            "extra": "mean: 229.93765300000177 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 4.093994494502808,
            "unit": "iter/sec",
            "range": "stddev: 0.022549654868522118",
            "extra": "mean: 244.26022099999045 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.9193461001744248,
            "unit": "iter/sec",
            "range": "stddev: 0.003355409456804257",
            "extra": "mean: 342.5424618000079 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.734739702410387,
            "unit": "iter/sec",
            "range": "stddev: 0.006352555910757801",
            "extra": "mean: 211.2048523999988 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 23.82254290320054,
            "unit": "iter/sec",
            "range": "stddev: 0.0006777712055521647",
            "extra": "mean: 41.97704686957037 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5576556437083172,
            "unit": "iter/sec",
            "range": "stddev: 0.022030838479037813",
            "extra": "mean: 1.7932213387999922 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 72.18744216381974,
            "unit": "iter/sec",
            "range": "stddev: 0.0003844230260983564",
            "extra": "mean: 13.852824951611856 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 100.00301830533493,
            "unit": "iter/sec",
            "range": "stddev: 0.0003626776476942919",
            "extra": "mean: 9.999698178576399 msec\nrounds: 84"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.2766593552982179,
            "unit": "iter/sec",
            "range": "stddev: 0.27566811148743414",
            "extra": "mean: 3.614553351799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 219.10100215925647,
            "unit": "iter/sec",
            "range": "stddev: 0.00022206928342218059",
            "extra": "mean: 4.5641050937463845 msec\nrounds: 64"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 10.09630185891089,
            "unit": "iter/sec",
            "range": "stddev: 0.0018401866260872475",
            "extra": "mean: 99.04616699999025 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.106497186539368,
            "unit": "iter/sec",
            "range": "stddev: 0.0023605510467883902",
            "extra": "mean: 98.94625027273337 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.158935819452244,
            "unit": "iter/sec",
            "range": "stddev: 0.0005988034211999094",
            "extra": "mean: 98.43550719999712 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.129473810798157,
            "unit": "iter/sec",
            "range": "stddev: 0.0009169790561260533",
            "extra": "mean: 98.72181109091632 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 10.036049306638219,
            "unit": "iter/sec",
            "range": "stddev: 0.0024439782978228357",
            "extra": "mean: 99.64080181815791 msec\nrounds: 11"
          }
        ]
      },
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
          "id": "32e8240b5e3a3c122407dff9992e794f9e83d85f",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/32e8240b5e3a3c122407dff9992e794f9e83d85f"
        },
        "date": 1773437731077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4824.738626780327,
            "unit": "iter/sec",
            "range": "stddev: 0.00030019824644681077",
            "extra": "mean: 207.26511368913805 usec\nrounds: 1293"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4311.349973313063,
            "unit": "iter/sec",
            "range": "stddev: 0.0002974824951710187",
            "extra": "mean: 231.94591164946618 usec\nrounds: 2558"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2184.0431445666118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000612170290624933",
            "extra": "mean: 457.86641279855934 usec\nrounds: 1422"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 1985.4120157280774,
            "unit": "iter/sec",
            "range": "stddev: 0.0005937923192454028",
            "extra": "mean: 503.67379268291904 usec\nrounds: 1476"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2025.6869240399726,
            "unit": "iter/sec",
            "range": "stddev: 0.0006289391319672919",
            "extra": "mean: 493.6597003872781 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2219.640920761728,
            "unit": "iter/sec",
            "range": "stddev: 0.000043812448991409766",
            "extra": "mean: 450.5233214284155 usec\nrounds: 2016"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2102.5251592030363,
            "unit": "iter/sec",
            "range": "stddev: 0.0012393315795825551",
            "extra": "mean: 475.61856542970014 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2259.5784851942644,
            "unit": "iter/sec",
            "range": "stddev: 0.00004354609626684158",
            "extra": "mean: 442.5604184817799 usec\nrounds: 1515"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2251.031371159682,
            "unit": "iter/sec",
            "range": "stddev: 0.00004382036837111373",
            "extra": "mean: 444.2408101513139 usec\nrounds: 2049"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2055.3987484054464,
            "unit": "iter/sec",
            "range": "stddev: 0.0018972451980399673",
            "extra": "mean: 486.5236007250603 usec\nrounds: 1931"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2255.3747785092505,
            "unit": "iter/sec",
            "range": "stddev: 0.00005200385179161432",
            "extra": "mean: 443.38528989890386 usec\nrounds: 1980"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2014.546058245234,
            "unit": "iter/sec",
            "range": "stddev: 0.0023295675502313803",
            "extra": "mean: 496.38974294340426 usec\nrounds: 1984"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2265.5914154535026,
            "unit": "iter/sec",
            "range": "stddev: 0.000042536760126535216",
            "extra": "mean: 441.38585323860366 usec\nrounds: 1976"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2288.99005545936,
            "unit": "iter/sec",
            "range": "stddev: 0.000051695855977851464",
            "extra": "mean: 436.87389449986824 usec\nrounds: 2000"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2292.1016684473184,
            "unit": "iter/sec",
            "range": "stddev: 0.00004301648290536903",
            "extra": "mean: 436.2808219922483 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2095.221007994151,
            "unit": "iter/sec",
            "range": "stddev: 0.0022683901045467978",
            "extra": "mean: 477.2766195950588 usec\nrounds: 2174"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2251.7459741927237,
            "unit": "iter/sec",
            "range": "stddev: 0.000052008563607858776",
            "extra": "mean: 444.09982807164175 usec\nrounds: 2059"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2265.1499562098443,
            "unit": "iter/sec",
            "range": "stddev: 0.000041309925254100076",
            "extra": "mean: 441.4718757398505 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2273.7611325010066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000402095110886547",
            "extra": "mean: 439.7999357566894 usec\nrounds: 2008"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 1972.319653165041,
            "unit": "iter/sec",
            "range": "stddev: 0.0032417478153213006",
            "extra": "mean: 507.01720605748153 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2319.2169535013536,
            "unit": "iter/sec",
            "range": "stddev: 0.00003631541174007866",
            "extra": "mean: 431.18001465550094 usec\nrounds: 2047"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2286.4393747451586,
            "unit": "iter/sec",
            "range": "stddev: 0.00003789526487825118",
            "extra": "mean: 437.3612574404943 usec\nrounds: 2016"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2291.9547410211294,
            "unit": "iter/sec",
            "range": "stddev: 0.00005442977904184548",
            "extra": "mean: 436.3087900917591 usec\nrounds: 1958"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2279.30924595763,
            "unit": "iter/sec",
            "range": "stddev: 0.00004005496375071163",
            "extra": "mean: 438.72940969879653 usec\nrounds: 1794"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 1911.4468852350633,
            "unit": "iter/sec",
            "range": "stddev: 0.0036147222122846533",
            "extra": "mean: 523.1638962738027 usec\nrounds: 1986"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2266.1059709721458,
            "unit": "iter/sec",
            "range": "stddev: 0.00003993824537644091",
            "extra": "mean: 441.2856295378835 usec\nrounds: 2011"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2262.4148204671546,
            "unit": "iter/sec",
            "range": "stddev: 0.000040496998448788965",
            "extra": "mean: 442.00559108497833 usec\nrounds: 2064"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2352.4665928097093,
            "unit": "iter/sec",
            "range": "stddev: 0.00004171191591202118",
            "extra": "mean: 425.0857389671292 usec\nrounds: 2130"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2341.2998296934497,
            "unit": "iter/sec",
            "range": "stddev: 0.00003995005689881147",
            "extra": "mean: 427.11317333967077 usec\nrounds: 2123"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2267.169825458863,
            "unit": "iter/sec",
            "range": "stddev: 0.00004154011883151585",
            "extra": "mean: 441.0785591668702 usec\nrounds: 1969"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 1785.9408974016276,
            "unit": "iter/sec",
            "range": "stddev: 0.005077065194910216",
            "extra": "mean: 559.9289435920886 usec\nrounds: 2021"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2238.529421835138,
            "unit": "iter/sec",
            "range": "stddev: 0.000044530429541313504",
            "extra": "mean: 446.72184794435435 usec\nrounds: 2019"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2295.71788712165,
            "unit": "iter/sec",
            "range": "stddev: 0.00003894236657948266",
            "extra": "mean: 435.5935917081653 usec\nrounds: 1592"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2206.8222393893775,
            "unit": "iter/sec",
            "range": "stddev: 0.00006951605445832629",
            "extra": "mean: 453.1402584907327 usec\nrounds: 2120"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2337.631548356058,
            "unit": "iter/sec",
            "range": "stddev: 0.00004153686878249774",
            "extra": "mean: 427.78341210497916 usec\nrounds: 1553"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2324.4290443790314,
            "unit": "iter/sec",
            "range": "stddev: 0.00004919952860262693",
            "extra": "mean: 430.2131753250179 usec\nrounds: 2002"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.4804266076197523,
            "unit": "iter/sec",
            "range": "stddev: 0.0028863133893435072",
            "extra": "mean: 403.15645580000137 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.17053272702862,
            "unit": "iter/sec",
            "range": "stddev: 0.0010259283238398095",
            "extra": "mean: 47.23546700000093 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1484266532023599,
            "unit": "iter/sec",
            "range": "stddev: 0.6555154364484184",
            "extra": "mean: 6.737334423600009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06632739063843696,
            "unit": "iter/sec",
            "range": "stddev: 0.7285172173444581",
            "extra": "mean: 15.076727583799993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.2319529340475235,
            "unit": "iter/sec",
            "range": "stddev: 0.004692480363304406",
            "extra": "mean: 236.29752400000825 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 17.640108672593403,
            "unit": "iter/sec",
            "range": "stddev: 0.0009543549597193186",
            "extra": "mean: 56.688993166672056 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.47869713209885184,
            "unit": "iter/sec",
            "range": "stddev: 0.03712965388103451",
            "extra": "mean: 2.08900353260002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 44.09381038587525,
            "unit": "iter/sec",
            "range": "stddev: 0.0006643579742889085",
            "extra": "mean: 22.678920039995774 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 99.89105455226732,
            "unit": "iter/sec",
            "range": "stddev: 0.0003890773239871075",
            "extra": "mean: 10.010906426828809 msec\nrounds: 82"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.787473594509203,
            "unit": "iter/sec",
            "range": "stddev: 0.0014425256294939258",
            "extra": "mean: 102.1714123000038 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 54.41236305302178,
            "unit": "iter/sec",
            "range": "stddev: 0.0017257737082263798",
            "extra": "mean: 18.3781762800038 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 22.35544737466416,
            "unit": "iter/sec",
            "range": "stddev: 0.0018060609556406744",
            "extra": "mean: 44.73182680000036 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.3134402992311336,
            "unit": "iter/sec",
            "range": "stddev: 0.020370494850108986",
            "extra": "mean: 3.190400221199991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.22978290104340185,
            "unit": "iter/sec",
            "range": "stddev: 1.1671629817939886",
            "extra": "mean: 4.351933914400002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5836702795610702,
            "unit": "iter/sec",
            "range": "stddev: 0.004547365727682051",
            "extra": "mean: 1.7132960765999883 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 20.988053837248792,
            "unit": "iter/sec",
            "range": "stddev: 0.0011006195060437368",
            "extra": "mean: 47.64615184211308 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 45.82276583436101,
            "unit": "iter/sec",
            "range": "stddev: 0.0008233200864731316",
            "extra": "mean: 21.823213457144316 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5245.782904072208,
            "unit": "iter/sec",
            "range": "stddev: 0.000020761766110273348",
            "extra": "mean: 190.62931468698747 usec\nrounds: 286"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9743899592301306,
            "unit": "iter/sec",
            "range": "stddev: 0.009157627199788763",
            "extra": "mean: 1.0262831534000043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 23.747346618622178,
            "unit": "iter/sec",
            "range": "stddev: 0.0012023197368991614",
            "extra": "mean: 42.10996773912504 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 108.12660526401395,
            "unit": "iter/sec",
            "range": "stddev: 0.0005072854761502281",
            "extra": "mean: 9.248417607843034 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 21.53257726813214,
            "unit": "iter/sec",
            "range": "stddev: 0.001393805079871339",
            "extra": "mean: 46.44125909999559 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 34.50477454171494,
            "unit": "iter/sec",
            "range": "stddev: 0.0008949801702769623",
            "extra": "mean: 28.98149642424235 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.4488619218002063,
            "unit": "iter/sec",
            "range": "stddev: 0.010054070477961185",
            "extra": "mean: 690.1968951999947 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.045691551664974,
            "unit": "iter/sec",
            "range": "stddev: 0.01589448244361407",
            "extra": "mean: 247.17653019999943 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.6710929499371163,
            "unit": "iter/sec",
            "range": "stddev: 0.028050907924453156",
            "extra": "mean: 272.39844199998515 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.8171820061270814,
            "unit": "iter/sec",
            "range": "stddev: 0.006169443792853755",
            "extra": "mean: 354.96464120000155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.566677481594362,
            "unit": "iter/sec",
            "range": "stddev: 0.00737369110364015",
            "extra": "mean: 218.97758359998534 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 22.58624839223301,
            "unit": "iter/sec",
            "range": "stddev: 0.0023153277732120957",
            "extra": "mean: 44.27472781818344 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.547723393771609,
            "unit": "iter/sec",
            "range": "stddev: 0.03192899134211009",
            "extra": "mean: 1.8257390708000003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 71.9310925829318,
            "unit": "iter/sec",
            "range": "stddev: 0.0006797125141950839",
            "extra": "mean: 13.90219394828552 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 97.84262195575225,
            "unit": "iter/sec",
            "range": "stddev: 0.0004731267611438543",
            "extra": "mean: 10.220494708862502 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.24991878832741787,
            "unit": "iter/sec",
            "range": "stddev: 1.2585229928029973",
            "extra": "mean: 4.001299808999965 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 217.78526521550523,
            "unit": "iter/sec",
            "range": "stddev: 0.0002778545367550752",
            "extra": "mean: 4.591678867762101 msec\nrounds: 121"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 8.694591547150994,
            "unit": "iter/sec",
            "range": "stddev: 0.00210573378276234",
            "extra": "mean: 115.01402850001341 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 8.621915364497534,
            "unit": "iter/sec",
            "range": "stddev: 0.0027916752617848217",
            "extra": "mean: 115.98350920002076 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 8.67511606199329,
            "unit": "iter/sec",
            "range": "stddev: 0.0013756830792933728",
            "extra": "mean: 115.27223299998468 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 8.626109132173184,
            "unit": "iter/sec",
            "range": "stddev: 0.0020815479934973694",
            "extra": "mean: 115.92712133333156 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 8.523460233807532,
            "unit": "iter/sec",
            "range": "stddev: 0.002032020496020668",
            "extra": "mean: 117.32324344443947 msec\nrounds: 9"
          }
        ]
      },
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
          "id": "c032b6b2db08edcc5a4cb1e0d2c6c01d2a480ba4",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/c032b6b2db08edcc5a4cb1e0d2c6c01d2a480ba4"
        },
        "date": 1773488412194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 5882.483918119154,
            "unit": "iter/sec",
            "range": "stddev: 0.0002855438348902856",
            "extra": "mean: 169.9962148506369 usec\nrounds: 1387"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 5176.93160826961,
            "unit": "iter/sec",
            "range": "stddev: 0.0003311967393883586",
            "extra": "mean: 193.16461480823966 usec\nrounds: 1864"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2741.287842516795,
            "unit": "iter/sec",
            "range": "stddev: 0.00004308943151882848",
            "extra": "mean: 364.7920457276362 usec\nrounds: 1487"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2674.720032905551,
            "unit": "iter/sec",
            "range": "stddev: 0.0003656037874400192",
            "extra": "mean: 373.8709052527263 usec\nrounds: 2132"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2444.792457074921,
            "unit": "iter/sec",
            "range": "stddev: 0.0004876487757321335",
            "extra": "mean: 409.03267559834217 usec\nrounds: 2090"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2686.2054072946844,
            "unit": "iter/sec",
            "range": "stddev: 0.000054159573302872956",
            "extra": "mean: 372.2723501651775 usec\nrounds: 2119"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2763.2012433065343,
            "unit": "iter/sec",
            "range": "stddev: 0.0007127121923847154",
            "extra": "mean: 361.89908441245785 usec\nrounds: 2085"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2721.199701858309,
            "unit": "iter/sec",
            "range": "stddev: 0.00006213177034800234",
            "extra": "mean: 367.4849733803438 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2602.5994891628693,
            "unit": "iter/sec",
            "range": "stddev: 0.0008998604646858596",
            "extra": "mean: 384.2312288786515 usec\nrounds: 2237"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2773.4315705962904,
            "unit": "iter/sec",
            "range": "stddev: 0.00004147054589341875",
            "extra": "mean: 360.5641511411075 usec\nrounds: 2104"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2765.1108227250534,
            "unit": "iter/sec",
            "range": "stddev: 0.00004904242448360929",
            "extra": "mean: 361.6491577051826 usec\nrounds: 2232"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2615.390215734398,
            "unit": "iter/sec",
            "range": "stddev: 0.0009681480218116412",
            "extra": "mean: 382.35212244196657 usec\nrounds: 2246"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2717.7050209423683,
            "unit": "iter/sec",
            "range": "stddev: 0.00005216824111219394",
            "extra": "mean: 367.95752014810216 usec\nrounds: 2184"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2831.9803363044552,
            "unit": "iter/sec",
            "range": "stddev: 0.00004026307024945592",
            "extra": "mean: 353.10979641367607 usec\nrounds: 2230"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2514.893172575606,
            "unit": "iter/sec",
            "range": "stddev: 0.0016116614861796683",
            "extra": "mean: 397.63120394329064 usec\nrounds: 1623"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2857.652154261663,
            "unit": "iter/sec",
            "range": "stddev: 0.00004448411632587723",
            "extra": "mean: 349.9376222220342 usec\nrounds: 2295"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2777.9886044237014,
            "unit": "iter/sec",
            "range": "stddev: 0.000041552397824047655",
            "extra": "mean: 359.97267894029096 usec\nrounds: 2227"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2517.6519747512325,
            "unit": "iter/sec",
            "range": "stddev: 0.0016426572825090206",
            "extra": "mean: 397.1954861230609 usec\nrounds: 2234"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2722.882005017321,
            "unit": "iter/sec",
            "range": "stddev: 0.000045613593482897936",
            "extra": "mean: 367.25792676926477 usec\nrounds: 2103"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2830.4260814680165,
            "unit": "iter/sec",
            "range": "stddev: 0.00004133746750312242",
            "extra": "mean: 353.3036974706452 usec\nrounds: 2056"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2792.877593066704,
            "unit": "iter/sec",
            "range": "stddev: 0.00004895691699257857",
            "extra": "mean: 358.05364419926315 usec\nrounds: 2077"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2767.752279926277,
            "unit": "iter/sec",
            "range": "stddev: 0.00005630946247031887",
            "extra": "mean: 361.3040109307168 usec\nrounds: 2104"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2446.5569964364495,
            "unit": "iter/sec",
            "range": "stddev: 0.002185617756249618",
            "extra": "mean: 408.7376674471747 usec\nrounds: 2135"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2738.385832705754,
            "unit": "iter/sec",
            "range": "stddev: 0.000049416159469987504",
            "extra": "mean: 365.178634820761 usec\nrounds: 2177"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2778.7312021019557,
            "unit": "iter/sec",
            "range": "stddev: 0.00004165049817937784",
            "extra": "mean: 359.87647860417576 usec\nrounds: 2150"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2771.3779717044636,
            "unit": "iter/sec",
            "range": "stddev: 0.00004241730131111706",
            "extra": "mean: 360.83133019382996 usec\nrounds: 2229"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2774.5107862254426,
            "unit": "iter/sec",
            "range": "stddev: 0.00004037459683216015",
            "extra": "mean: 360.42390066194 usec\nrounds: 2114"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2882.6471293930836,
            "unit": "iter/sec",
            "range": "stddev: 0.00003983338479826644",
            "extra": "mean: 346.90336871392975 usec\nrounds: 2148"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2405.9263720879235,
            "unit": "iter/sec",
            "range": "stddev: 0.0028168675268028923",
            "extra": "mean: 415.64031701110406 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2753.2625880663895,
            "unit": "iter/sec",
            "range": "stddev: 0.000044424064531181546",
            "extra": "mean: 363.2054582568159 usec\nrounds: 1569"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2742.7702656377564,
            "unit": "iter/sec",
            "range": "stddev: 0.00004431309981404831",
            "extra": "mean: 364.5948815065914 usec\nrounds: 2152"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2753.301688411116,
            "unit": "iter/sec",
            "range": "stddev: 0.00004503021374012103",
            "extra": "mean: 363.20030028277904 usec\nrounds: 2128"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2815.791554196731,
            "unit": "iter/sec",
            "range": "stddev: 0.00004212727494686848",
            "extra": "mean: 355.1399245123714 usec\nrounds: 2252"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2818.4927142993265,
            "unit": "iter/sec",
            "range": "stddev: 0.00003913397484104366",
            "extra": "mean: 354.79956890667313 usec\nrounds: 2097"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2875.8704380614213,
            "unit": "iter/sec",
            "range": "stddev: 0.00004134908396610294",
            "extra": "mean: 347.7208106336265 usec\nrounds: 2144"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2869.675652667511,
            "unit": "iter/sec",
            "range": "stddev: 0.00004268121103693937",
            "extra": "mean: 348.47143755443193 usec\nrounds: 2242"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 3.0659345561605313,
            "unit": "iter/sec",
            "range": "stddev: 0.003382121687523894",
            "extra": "mean: 326.16482239996003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 22.364180598002896,
            "unit": "iter/sec",
            "range": "stddev: 0.0007408690331094543",
            "extra": "mean: 44.71435900000285 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1620078741278712,
            "unit": "iter/sec",
            "range": "stddev: 0.17451803869482116",
            "extra": "mean: 6.172539485400011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.07104456557091435,
            "unit": "iter/sec",
            "range": "stddev: 0.6555228837461538",
            "extra": "mean: 14.07567196679995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.090790189117265,
            "unit": "iter/sec",
            "range": "stddev: 0.009709175525012787",
            "extra": "mean: 244.45154939999156 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 21.9150431440056,
            "unit": "iter/sec",
            "range": "stddev: 0.002965748407967307",
            "extra": "mean: 45.63075661904544 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5277143464781503,
            "unit": "iter/sec",
            "range": "stddev: 0.009928683748310485",
            "extra": "mean: 1.8949645895999994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 46.911124660811296,
            "unit": "iter/sec",
            "range": "stddev: 0.0004955921875342872",
            "extra": "mean: 21.3169052592632 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 107.73936150493071,
            "unit": "iter/sec",
            "range": "stddev: 0.00036059312747746926",
            "extra": "mean: 9.281658866655107 msec\nrounds: 90"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 11.899248680645037,
            "unit": "iter/sec",
            "range": "stddev: 0.0006148002208165561",
            "extra": "mean: 84.03891933333323 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 62.47762213351101,
            "unit": "iter/sec",
            "range": "stddev: 0.00044309524648945896",
            "extra": "mean: 16.0057307857053 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 27.875805836591514,
            "unit": "iter/sec",
            "range": "stddev: 0.00036727365908044044",
            "extra": "mean: 35.87340239998866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.39973400362688194,
            "unit": "iter/sec",
            "range": "stddev: 0.012352581112585307",
            "extra": "mean: 2.501663583599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.2888922853145947,
            "unit": "iter/sec",
            "range": "stddev: 0.7914153139876873",
            "extra": "mean: 3.4614977652000336 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.7070742833667532,
            "unit": "iter/sec",
            "range": "stddev: 0.013584602474025027",
            "extra": "mean: 1.4142785609999464 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 26.577534092571423,
            "unit": "iter/sec",
            "range": "stddev: 0.0007430882295357158",
            "extra": "mean: 37.62576304170769 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 51.98141449745922,
            "unit": "iter/sec",
            "range": "stddev: 0.0004416446369591339",
            "extra": "mean: 19.237645025009442 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 7255.36759953387,
            "unit": "iter/sec",
            "range": "stddev: 0.000018627594139838846",
            "extra": "mean: 137.82899160950112 usec\nrounds: 477"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9813654396997321,
            "unit": "iter/sec",
            "range": "stddev: 0.009017791118528656",
            "extra": "mean: 1.0189884007999808 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 26.979653933917398,
            "unit": "iter/sec",
            "range": "stddev: 0.0006892753118707713",
            "extra": "mean: 37.06496764003532 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 115.37868368141386,
            "unit": "iter/sec",
            "range": "stddev: 0.0002851622366878574",
            "extra": "mean: 8.66711222639029 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 24.161980420774785,
            "unit": "iter/sec",
            "range": "stddev: 0.0006488485653773051",
            "extra": "mean: 41.38733591308546 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 41.726088659112065,
            "unit": "iter/sec",
            "range": "stddev: 0.0004954714797326071",
            "extra": "mean: 23.96582167501151 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.7384086148737017,
            "unit": "iter/sec",
            "range": "stddev: 0.004013591634562194",
            "extra": "mean: 575.2387508000538 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.881251324238832,
            "unit": "iter/sec",
            "range": "stddev: 0.0009776981563630303",
            "extra": "mean: 204.86550140008148 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.9623634621256727,
            "unit": "iter/sec",
            "range": "stddev: 0.020267012569995236",
            "extra": "mean: 252.37462680001954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 3.389730609091936,
            "unit": "iter/sec",
            "range": "stddev: 0.009514881030475358",
            "extra": "mean: 295.0086939999892 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 5.812198569357368,
            "unit": "iter/sec",
            "range": "stddev: 0.0020566639650205416",
            "extra": "mean: 172.05193319996397 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 27.48907212518254,
            "unit": "iter/sec",
            "range": "stddev: 0.0005858137878089927",
            "extra": "mean: 36.37809219045656 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5718822965682993,
            "unit": "iter/sec",
            "range": "stddev: 0.04168163253940347",
            "extra": "mean: 1.7486115692000113 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 77.39318369480986,
            "unit": "iter/sec",
            "range": "stddev: 0.0006666940549626576",
            "extra": "mean: 12.921034544119186 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 106.74709513637441,
            "unit": "iter/sec",
            "range": "stddev: 0.0003096396769954448",
            "extra": "mean: 9.36793641759013 msec\nrounds: 91"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.2695117067287398,
            "unit": "iter/sec",
            "range": "stddev: 1.0511799954179153",
            "extra": "mean: 3.7104139636000584 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 234.89631253424164,
            "unit": "iter/sec",
            "range": "stddev: 0.000221095900842206",
            "extra": "mean: 4.257197523499763 msec\nrounds: 149"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 9.5859076957403,
            "unit": "iter/sec",
            "range": "stddev: 0.0025143477822005544",
            "extra": "mean: 104.31980275006936 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 9.562769777929706,
            "unit": "iter/sec",
            "range": "stddev: 0.00300073899242287",
            "extra": "mean: 104.57221319998098 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 9.629198747238942,
            "unit": "iter/sec",
            "range": "stddev: 0.0009214474180665754",
            "extra": "mean: 103.85080069997912 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 9.61172571791088,
            "unit": "iter/sec",
            "range": "stddev: 0.0012113532937702684",
            "extra": "mean: 104.03958969995983 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.675303865253255,
            "unit": "iter/sec",
            "range": "stddev: 0.000749715927459574",
            "extra": "mean: 103.35592700000689 msec\nrounds: 10"
          }
        ]
      },
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
          "id": "33908f83d2d0b0c34463e89e14d74aca8b701cee",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/33908f83d2d0b0c34463e89e14d74aca8b701cee"
        },
        "date": 1773528121471,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4641.100233915484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003933277519020511",
            "extra": "mean: 215.46615017972707 usec\nrounds: 1112"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4168.653826009174,
            "unit": "iter/sec",
            "range": "stddev: 0.000481353090151097",
            "extra": "mean: 239.88559418409224 usec\nrounds: 2304"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2192.662315271346,
            "unit": "iter/sec",
            "range": "stddev: 0.00004180185745666821",
            "extra": "mean: 456.0665785311534 usec\nrounds: 885"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2098.3610130339825,
            "unit": "iter/sec",
            "range": "stddev: 0.000504172113415984",
            "extra": "mean: 476.5624188538072 usec\nrounds: 1867"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2177.658030989665,
            "unit": "iter/sec",
            "range": "stddev: 0.000051289860679941836",
            "extra": "mean: 459.20892342565696 usec\nrounds: 1985"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2069.8730319033116,
            "unit": "iter/sec",
            "range": "stddev: 0.0007324755683149171",
            "extra": "mean: 483.1214207764567 usec\nrounds: 1906"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2296.990275763545,
            "unit": "iter/sec",
            "range": "stddev: 0.000044364526770121003",
            "extra": "mean: 435.3523001605172 usec\nrounds: 1869"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2070.0520178961756,
            "unit": "iter/sec",
            "range": "stddev: 0.0013351553089246792",
            "extra": "mean: 483.07964792899975 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2207.1875072483135,
            "unit": "iter/sec",
            "range": "stddev: 0.00004099152565048651",
            "extra": "mean: 453.0652682275706 usec\nrounds: 1495"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 1897.447287617845,
            "unit": "iter/sec",
            "range": "stddev: 0.0018286590231692751",
            "extra": "mean: 527.0238633377018 usec\nrounds: 1522"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2205.6373701548514,
            "unit": "iter/sec",
            "range": "stddev: 0.00004605289629091325",
            "extra": "mean: 453.3836856100207 usec\nrounds: 1918"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2219.084848266119,
            "unit": "iter/sec",
            "range": "stddev: 0.00004003480866458472",
            "extra": "mean: 450.6362164481225 usec\nrounds: 1982"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 1975.2302232567797,
            "unit": "iter/sec",
            "range": "stddev: 0.0020860760404388047",
            "extra": "mean: 506.2700986577604 usec\nrounds: 1490"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2257.316781859634,
            "unit": "iter/sec",
            "range": "stddev: 0.000041743460065961486",
            "extra": "mean: 443.00383891009534 usec\nrounds: 2092"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2266.0413127701718,
            "unit": "iter/sec",
            "range": "stddev: 0.00003856191865249144",
            "extra": "mean: 441.29822098323893 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2132.325668728071,
            "unit": "iter/sec",
            "range": "stddev: 0.00011245279950356918",
            "extra": "mean: 468.9715153110259 usec\nrounds: 2090"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 1974.3248895530676,
            "unit": "iter/sec",
            "range": "stddev: 0.002311123035435445",
            "extra": "mean: 506.50225061305497 usec\nrounds: 2039"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2214.830118851226,
            "unit": "iter/sec",
            "range": "stddev: 0.00004140654532766624",
            "extra": "mean: 451.5018969123797 usec\nrounds: 2008"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2221.243594217615,
            "unit": "iter/sec",
            "range": "stddev: 0.0000393566182180711",
            "extra": "mean: 450.19825948095917 usec\nrounds: 2004"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2251.418397369052,
            "unit": "iter/sec",
            "range": "stddev: 0.00005025942136714866",
            "extra": "mean: 444.164443698503 usec\nrounds: 1936"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 1953.6045921926757,
            "unit": "iter/sec",
            "range": "stddev: 0.0030217528718749563",
            "extra": "mean: 511.8743086479059 usec\nrounds: 2093"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2226.628263266703,
            "unit": "iter/sec",
            "range": "stddev: 0.00004196037947702572",
            "extra": "mean: 449.1095422155886 usec\nrounds: 2049"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2224.4484434430406,
            "unit": "iter/sec",
            "range": "stddev: 0.00004127298305663603",
            "extra": "mean: 449.54964137185505 usec\nrounds: 1779"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2215.8436563183504,
            "unit": "iter/sec",
            "range": "stddev: 0.00005821931743034534",
            "extra": "mean: 451.2953777892035 usec\nrounds: 1927"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2214.8169792840895,
            "unit": "iter/sec",
            "range": "stddev: 0.00003962619473969521",
            "extra": "mean: 451.5045754811022 usec\nrounds: 1974"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 1827.3117149668024,
            "unit": "iter/sec",
            "range": "stddev: 0.004063992602165514",
            "extra": "mean: 547.2520051228192 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2199.813407880667,
            "unit": "iter/sec",
            "range": "stddev: 0.000049836774515508364",
            "extra": "mean: 454.5840099062833 usec\nrounds: 1918"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2280.404407525593,
            "unit": "iter/sec",
            "range": "stddev: 0.00004262220978142552",
            "extra": "mean: 438.51871040938477 usec\nrounds: 2027"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2293.4315990027508,
            "unit": "iter/sec",
            "range": "stddev: 0.000042005189112108326",
            "extra": "mean: 436.02782853206895 usec\nrounds: 1458"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2210.081677962206,
            "unit": "iter/sec",
            "range": "stddev: 0.00004481676023637239",
            "extra": "mean: 452.4719651637693 usec\nrounds: 1464"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2207.7310965205315,
            "unit": "iter/sec",
            "range": "stddev: 0.00004209571705979123",
            "extra": "mean: 452.95371414391826 usec\nrounds: 2015"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2189.7410849476946,
            "unit": "iter/sec",
            "range": "stddev: 0.000054895054108876487",
            "extra": "mean: 456.67499544764064 usec\nrounds: 1977"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 1709.0290739729853,
            "unit": "iter/sec",
            "range": "stddev: 0.005356509037284229",
            "extra": "mean: 585.1275529650861 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2236.446047934655,
            "unit": "iter/sec",
            "range": "stddev: 0.000052213642421103046",
            "extra": "mean: 447.13799419552026 usec\nrounds: 1206"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2253.898257906761,
            "unit": "iter/sec",
            "range": "stddev: 0.00005745053852531332",
            "extra": "mean: 443.675749999789 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2288.8548875673932,
            "unit": "iter/sec",
            "range": "stddev: 0.00004241626099544071",
            "extra": "mean: 436.89969400498126 usec\nrounds: 2085"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.097466456465907,
            "unit": "iter/sec",
            "range": "stddev: 0.1679782591293536",
            "extra": "mean: 476.7656697999996 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.596181475490795,
            "unit": "iter/sec",
            "range": "stddev: 0.0012128620554439278",
            "extra": "mean: 46.30448216666849 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.14952680767973536,
            "unit": "iter/sec",
            "range": "stddev: 0.6081301746869485",
            "extra": "mean: 6.687763990399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06527268616903809,
            "unit": "iter/sec",
            "range": "stddev: 0.752767468760974",
            "extra": "mean: 15.320343909399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.19845613799885,
            "unit": "iter/sec",
            "range": "stddev: 0.009919588406543384",
            "extra": "mean: 238.18279080001048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 17.75666379077163,
            "unit": "iter/sec",
            "range": "stddev: 0.001768616353586382",
            "extra": "mean: 56.3168854117581 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.4609820125765575,
            "unit": "iter/sec",
            "range": "stddev: 0.011179543164810811",
            "extra": "mean: 2.169282038599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 42.487643902728465,
            "unit": "iter/sec",
            "range": "stddev: 0.0007801776203396238",
            "extra": "mean: 23.53625450000023 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 98.56001917583534,
            "unit": "iter/sec",
            "range": "stddev: 0.00042570067889074716",
            "extra": "mean: 10.14610192207813 msec\nrounds: 77"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.846270678062641,
            "unit": "iter/sec",
            "range": "stddev: 0.001984372508004558",
            "extra": "mean: 101.5612949000058 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.24988798028219,
            "unit": "iter/sec",
            "range": "stddev: 0.0005372564438041951",
            "extra": "mean: 18.099584208331503 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 23.079072058088567,
            "unit": "iter/sec",
            "range": "stddev: 0.00197214876468975",
            "extra": "mean: 43.32929839999906 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.31305639566028987,
            "unit": "iter/sec",
            "range": "stddev: 0.07239804518980675",
            "extra": "mean: 3.1943126345999984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.22553014117352926,
            "unit": "iter/sec",
            "range": "stddev: 1.1282406733394226",
            "extra": "mean: 4.433997135800007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5803559159381043,
            "unit": "iter/sec",
            "range": "stddev: 0.01510511227065159",
            "extra": "mean: 1.7230805657999895 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.808192119792427,
            "unit": "iter/sec",
            "range": "stddev: 0.001030420148121907",
            "extra": "mean: 45.854328249998844 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 45.57668954386058,
            "unit": "iter/sec",
            "range": "stddev: 0.0008201410758019869",
            "extra": "mean: 21.941040694446517 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5197.755474198084,
            "unit": "iter/sec",
            "range": "stddev: 0.000013277301102504516",
            "extra": "mean: 192.3907357635521 usec\nrounds: 439"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9506715883680155,
            "unit": "iter/sec",
            "range": "stddev: 0.007673718638881754",
            "extra": "mean: 1.0518879623999964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 23.29246992403332,
            "unit": "iter/sec",
            "range": "stddev: 0.0009111497702063712",
            "extra": "mean: 42.93232977273027 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 104.02269704168582,
            "unit": "iter/sec",
            "range": "stddev: 0.0007825058229276913",
            "extra": "mean: 9.613286604165458 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 20.967289769922072,
            "unit": "iter/sec",
            "range": "stddev: 0.0012964577477215585",
            "extra": "mean: 47.693336190474966 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 33.02034558499573,
            "unit": "iter/sec",
            "range": "stddev: 0.0008463976241850816",
            "extra": "mean: 30.284358999997707 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.3974369114913836,
            "unit": "iter/sec",
            "range": "stddev: 0.008404910990562826",
            "extra": "mean: 715.5958109999915 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 3.900298461446903,
            "unit": "iter/sec",
            "range": "stddev: 0.013658307977137388",
            "extra": "mean: 256.39063520001173 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.6798102920609215,
            "unit": "iter/sec",
            "range": "stddev: 0.02598475850314342",
            "extra": "mean: 271.75313960001404 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.721861485511928,
            "unit": "iter/sec",
            "range": "stddev: 0.006016688706278856",
            "extra": "mean: 367.39562440001237 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.512969628001999,
            "unit": "iter/sec",
            "range": "stddev: 0.005255400586834181",
            "extra": "mean: 221.5835874000163 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 22.539145488436322,
            "unit": "iter/sec",
            "range": "stddev: 0.0010230695383497948",
            "extra": "mean: 44.367254318183825 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5367929333274206,
            "unit": "iter/sec",
            "range": "stddev: 0.04742647938980531",
            "extra": "mean: 1.862915731399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 68.03330171612244,
            "unit": "iter/sec",
            "range": "stddev: 0.0008707609940172944",
            "extra": "mean: 14.698683949995939 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 98.3685178248088,
            "unit": "iter/sec",
            "range": "stddev: 0.00035645445328244367",
            "extra": "mean: 10.165854097557595 msec\nrounds: 82"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.24315303615991338,
            "unit": "iter/sec",
            "range": "stddev: 1.952245158066625",
            "extra": "mean: 4.112636287799978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 189.1534354638956,
            "unit": "iter/sec",
            "range": "stddev: 0.00042743403143013364",
            "extra": "mean: 5.2867133898335865 msec\nrounds: 118"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 7.830172074405297,
            "unit": "iter/sec",
            "range": "stddev: 0.006484990222878254",
            "extra": "mean: 127.7111141999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 8.12855090284788,
            "unit": "iter/sec",
            "range": "stddev: 0.0017236627826645217",
            "extra": "mean: 123.02315775000494 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 8.05939576958364,
            "unit": "iter/sec",
            "range": "stddev: 0.010433673658619946",
            "extra": "mean: 124.07878066666298 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 7.99281572153147,
            "unit": "iter/sec",
            "range": "stddev: 0.0028715756816835815",
            "extra": "mean: 125.1123552499962 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 8.095772003209124,
            "unit": "iter/sec",
            "range": "stddev: 0.0019935572874956492",
            "extra": "mean: 123.52126512500661 msec\nrounds: 8"
          }
        ]
      },
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
          "id": "02f79b0c7b4c13b4f6a74ab3d1ac717f598d2e47",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/02f79b0c7b4c13b4f6a74ab3d1ac717f598d2e47"
        },
        "date": 1773835824416,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 7201.161944722673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000831096969436392",
            "extra": "mean: 138.8664784483626 usec\nrounds: 1392"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 6433.662872326401,
            "unit": "iter/sec",
            "range": "stddev: 0.00005093847430049002",
            "extra": "mean: 155.4324526859148 usec\nrounds: 2737"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2610.3859983781535,
            "unit": "iter/sec",
            "range": "stddev: 0.0007087555705422343",
            "extra": "mean: 383.0851071915437 usec\nrounds: 1474"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2724.3588310637547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000684031984827373",
            "extra": "mean: 367.0588428358901 usec\nrounds: 1368"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2657.908134764951,
            "unit": "iter/sec",
            "range": "stddev: 0.00007833197054816737",
            "extra": "mean: 376.2357272323236 usec\nrounds: 2262"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2733.751751719283,
            "unit": "iter/sec",
            "range": "stddev: 0.00005758111184732659",
            "extra": "mean: 365.79766226801326 usec\nrounds: 2425"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2772.3786165978854,
            "unit": "iter/sec",
            "range": "stddev: 0.0009464669278868619",
            "extra": "mean: 360.7010940039447 usec\nrounds: 2585"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2826.4434462777317,
            "unit": "iter/sec",
            "range": "stddev: 0.00005260423323361256",
            "extra": "mean: 353.80152442708317 usec\nrounds: 2313"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2764.6376209343525,
            "unit": "iter/sec",
            "range": "stddev: 0.00006376215165140249",
            "extra": "mean: 361.7110584142432 usec\nrounds: 2157"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2573.0551226764233,
            "unit": "iter/sec",
            "range": "stddev: 0.00126472998295261",
            "extra": "mean: 388.6430536162889 usec\nrounds: 2406"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2782.5816786632095,
            "unit": "iter/sec",
            "range": "stddev: 0.00006243309834457954",
            "extra": "mean: 359.3784892885566 usec\nrounds: 2334"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2776.3931999830947,
            "unit": "iter/sec",
            "range": "stddev: 0.00006874365828428959",
            "extra": "mean: 360.1795307689447 usec\nrounds: 2470"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2251.3101481776575,
            "unit": "iter/sec",
            "range": "stddev: 0.0024645015132261617",
            "extra": "mean: 444.185800348059 usec\nrounds: 2299"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2107.3405665742166,
            "unit": "iter/sec",
            "range": "stddev: 0.00009295536322658613",
            "extra": "mean: 474.53174672456623 usec\nrounds: 1832"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2295.195898836796,
            "unit": "iter/sec",
            "range": "stddev: 0.00009195116696321456",
            "extra": "mean: 435.69265721797404 usec\nrounds: 1905"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2870.843101638143,
            "unit": "iter/sec",
            "range": "stddev: 0.00007553494836631076",
            "extra": "mean: 348.3297291410269 usec\nrounds: 2433"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2322.104648910883,
            "unit": "iter/sec",
            "range": "stddev: 0.0030388187287256874",
            "extra": "mean: 430.6438129173125 usec\nrounds: 2245"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2746.2968714688936,
            "unit": "iter/sec",
            "range": "stddev: 0.00007547134201545443",
            "extra": "mean: 364.12669379954417 usec\nrounds: 2371"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2796.4917383668153,
            "unit": "iter/sec",
            "range": "stddev: 0.00006222379679234353",
            "extra": "mean: 357.59090087067875 usec\nrounds: 2411"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2823.5685530424976,
            "unit": "iter/sec",
            "range": "stddev: 0.00007155647836274132",
            "extra": "mean: 354.1617570866001 usec\nrounds: 2293"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2300.87804472933,
            "unit": "iter/sec",
            "range": "stddev: 0.0037952965474958985",
            "extra": "mean: 434.61669004609826 usec\nrounds: 2381"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2739.3660992934347,
            "unit": "iter/sec",
            "range": "stddev: 0.00006873916251264763",
            "extra": "mean: 365.04795772201834 usec\nrounds: 2318"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2781.558181652715,
            "unit": "iter/sec",
            "range": "stddev: 0.000051553416625263544",
            "extra": "mean: 359.51072553364 usec\nrounds: 1359"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2769.8286476988224,
            "unit": "iter/sec",
            "range": "stddev: 0.000053347834509706654",
            "extra": "mean: 361.03316384961266 usec\nrounds: 2307"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2802.930742592197,
            "unit": "iter/sec",
            "range": "stddev: 0.00005758210533139569",
            "extra": "mean: 356.7694287997938 usec\nrounds: 2500"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2729.1200335178455,
            "unit": "iter/sec",
            "range": "stddev: 0.00006517262127267917",
            "extra": "mean: 366.4184747165541 usec\nrounds: 2294"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2158.4093089587827,
            "unit": "iter/sec",
            "range": "stddev: 0.004628870236044342",
            "extra": "mean: 463.30415452220245 usec\nrounds: 2388"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2790.191008148854,
            "unit": "iter/sec",
            "range": "stddev: 0.0000791932915752346",
            "extra": "mean: 358.3984025034357 usec\nrounds: 2477"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2750.833284521501,
            "unit": "iter/sec",
            "range": "stddev: 0.00008392620637136944",
            "extra": "mean: 363.52621063109865 usec\nrounds: 2502"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2653.440449741519,
            "unit": "iter/sec",
            "range": "stddev: 0.00009383315982877376",
            "extra": "mean: 376.86920771009335 usec\nrounds: 2205"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2805.466393131524,
            "unit": "iter/sec",
            "range": "stddev: 0.00005058811410044449",
            "extra": "mean: 356.4469716865073 usec\nrounds: 2437"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2805.1259942548772,
            "unit": "iter/sec",
            "range": "stddev: 0.00005412555273120894",
            "extra": "mean: 356.49022612463045 usec\nrounds: 2335"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2875.438922387757,
            "unit": "iter/sec",
            "range": "stddev: 0.00005355985139386559",
            "extra": "mean: 347.7729929208869 usec\nrounds: 2260"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2050.417256852496,
            "unit": "iter/sec",
            "range": "stddev: 0.005696624378708565",
            "extra": "mean: 487.7056104839146 usec\nrounds: 2480"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2835.7272894085313,
            "unit": "iter/sec",
            "range": "stddev: 0.00006644422358795287",
            "extra": "mean: 352.64321916109833 usec\nrounds: 2359"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2882.7778605429303,
            "unit": "iter/sec",
            "range": "stddev: 0.000059853895893022824",
            "extra": "mean: 346.887636986245 usec\nrounds: 1314"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 3.0865977049706195,
            "unit": "iter/sec",
            "range": "stddev: 0.005687251582755123",
            "extra": "mean: 323.9813204000029 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 19.022115684473,
            "unit": "iter/sec",
            "range": "stddev: 0.0009205856312199266",
            "extra": "mean: 52.570387888885584 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.20373247290579694,
            "unit": "iter/sec",
            "range": "stddev: 0.2137800729397918",
            "extra": "mean: 4.908397693000007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.07185530380985752,
            "unit": "iter/sec",
            "range": "stddev: 0.7382819680123313",
            "extra": "mean: 13.916857169599973 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.707494806152861,
            "unit": "iter/sec",
            "range": "stddev: 0.015709038576891605",
            "extra": "mean: 212.4272125999937 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 23.5144502236575,
            "unit": "iter/sec",
            "range": "stddev: 0.002826395582069157",
            "extra": "mean: 42.527041478261594 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5906890475539672,
            "unit": "iter/sec",
            "range": "stddev: 0.11303014915948079",
            "extra": "mean: 1.6929381104000185 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 57.17244132889756,
            "unit": "iter/sec",
            "range": "stddev: 0.0020427229635388657",
            "extra": "mean: 17.490944531251884 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 126.53497911088377,
            "unit": "iter/sec",
            "range": "stddev: 0.001339353152789868",
            "extra": "mean: 7.902953057143912 msec\nrounds: 105"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 13.512719289090995,
            "unit": "iter/sec",
            "range": "stddev: 0.001502388684828886",
            "extra": "mean: 74.00434942856496 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 75.91726657756978,
            "unit": "iter/sec",
            "range": "stddev: 0.0009121734399514782",
            "extra": "mean: 13.172233999998312 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 28.92060638785,
            "unit": "iter/sec",
            "range": "stddev: 0.0016703180794082337",
            "extra": "mean: 34.577421599988156 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.43634264695583486,
            "unit": "iter/sec",
            "range": "stddev: 0.06429644339709514",
            "extra": "mean: 2.291776902800007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.35611250579036774,
            "unit": "iter/sec",
            "range": "stddev: 0.8902139908869253",
            "extra": "mean: 2.8081013267999877 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.7727237147349802,
            "unit": "iter/sec",
            "range": "stddev: 0.015350898483716647",
            "extra": "mean: 1.2941236057999959 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 27.339695758072658,
            "unit": "iter/sec",
            "range": "stddev: 0.00155033981958414",
            "extra": "mean: 36.57685179999589 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 55.967803670227696,
            "unit": "iter/sec",
            "range": "stddev: 0.0011477379957230836",
            "extra": "mean: 17.867415450000124 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 7294.138423060665,
            "unit": "iter/sec",
            "range": "stddev: 0.000017690685318137043",
            "extra": "mean: 137.09638369878837 usec\nrounds: 503"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 1.0039521592749403,
            "unit": "iter/sec",
            "range": "stddev: 0.008561971929599682",
            "extra": "mean: 996.063398800004 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 33.16372952959423,
            "unit": "iter/sec",
            "range": "stddev: 0.0011690170992469683",
            "extra": "mean: 30.153424062502765 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 142.67559519068172,
            "unit": "iter/sec",
            "range": "stddev: 0.0007063398530021591",
            "extra": "mean: 7.008907155169246 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 29.048754643238528,
            "unit": "iter/sec",
            "range": "stddev: 0.0014414731191906918",
            "extra": "mean: 34.42488369231219 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 45.388977302814105,
            "unit": "iter/sec",
            "range": "stddev: 0.0011368838079847052",
            "extra": "mean: 22.03178082926315 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.9071034514651897,
            "unit": "iter/sec",
            "range": "stddev: 0.007218918622246895",
            "extra": "mean: 524.3554036000091 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 5.21549170979339,
            "unit": "iter/sec",
            "range": "stddev: 0.0046335469010062615",
            "extra": "mean: 191.73647580001898 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 4.699138260278104,
            "unit": "iter/sec",
            "range": "stddev: 0.008413082667270012",
            "extra": "mean: 212.80497499999456 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 3.7032670007433772,
            "unit": "iter/sec",
            "range": "stddev: 0.004128140972156631",
            "extra": "mean: 270.0318393999851 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 5.580673117329737,
            "unit": "iter/sec",
            "range": "stddev: 0.005045006531108903",
            "extra": "mean: 179.18985380001686 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 30.44980076086267,
            "unit": "iter/sec",
            "range": "stddev: 0.001500922361466235",
            "extra": "mean: 32.84093737931141 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.43246881030615214,
            "unit": "iter/sec",
            "range": "stddev: 0.041714827440866145",
            "extra": "mean: 2.312305480000009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 95.4047198084448,
            "unit": "iter/sec",
            "range": "stddev: 0.0009556290433557664",
            "extra": "mean: 10.481661724994495 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 130.79421151243073,
            "unit": "iter/sec",
            "range": "stddev: 0.0009320944423302484",
            "extra": "mean: 7.645598290907236 msec\nrounds: 110"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.20218109468834308,
            "unit": "iter/sec",
            "range": "stddev: 3.0555855858014533",
            "extra": "mean: 4.946060864599997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 286.06387978738894,
            "unit": "iter/sec",
            "range": "stddev: 0.0005703252797483483",
            "extra": "mean: 3.495722706212435 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 10.660133014288741,
            "unit": "iter/sec",
            "range": "stddev: 0.001330726122547023",
            "extra": "mean: 93.80745987499495 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.683082770773447,
            "unit": "iter/sec",
            "range": "stddev: 0.0017625021447320662",
            "extra": "mean: 93.60593954544458 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.537209055359943,
            "unit": "iter/sec",
            "range": "stddev: 0.0018209923658335629",
            "extra": "mean: 94.90178990909665 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.491832969794254,
            "unit": "iter/sec",
            "range": "stddev: 0.0018475607097458938",
            "extra": "mean: 95.31223027272517 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 10.565497720394328,
            "unit": "iter/sec",
            "range": "stddev: 0.0015426608730125426",
            "extra": "mean: 94.64769445453798 msec\nrounds: 11"
          }
        ]
      }
    ]
  }
}