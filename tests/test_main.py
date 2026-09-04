def test_main(capsys):
    from rail_defect_detector import main

    main()
    captured = capsys.readouterr()
    assert "rail-defect-detector" in captured.out
