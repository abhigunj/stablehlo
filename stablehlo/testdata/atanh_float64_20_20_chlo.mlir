// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf64>
    %1 = call @expected() : () -> tensor<20x20xf64>
    %2 = chlo.atanh %0 : tensor<20x20xf64> -> tensor<20x20xf64>
    stablehlo.custom_call @check.expect_almost_eq(%2, %1) {has_side_effect = true} : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    return %2 : tensor<20x20xf64>
  }
  func.func private @inputs() -> (tensor<20x20xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7EEB29BB60791C40CC832AE1746B0740588D1D5C0A8AF73FB3AFE9EF10D504C087AA1267940FF13F08014E60440913C0861FBC89518CD93F3ABC8A8252C304C08A42E81705880BC04A26DE65252F0B40B69A8A83167FDCBFEECC1EE4045504C0684C7BE069A90C404F305CF8E9F2F1BFF47733F9D20D0BC078CFF8041B611240FC3D4CE3D3EDF9BFC5787F14B2ACE73F44DD37014DE5FBBF4E10ACB7AF1B0440D0DA12396D37DFBF5082209DD48DEDBFED6E89CFAC45F13FD2F68C45F9C520401F8AF63E9A8FE63F4C10389C73C1F3BFDAC06F6DB9E4E1BFC64BF3E8CCA00640878AD724B8C5F73F8F641C22256F1140BCB129D940ACEA3F5584AAB2848DF23F8F57E9578596D1BF18BB8D9157450B40D6C84986EC9D054044FC4FEFF566F83F3D93B4DB464002408E8DAE417AF8E5BF6F5A20ED08370F40402181D88F23F23FBC5489AB3B44EF3F60FF0926CC1BB2BF509627CF779509C0CE19E0F850A60DC03AE8AD5510CBE5BFDB7520CD78F000C0AEF5BFC56D73FEBFFDE442FB9D2314C02CEAB5A3A0B40F4052AA1CE00C910040D66FA12A4E1B0B40FC1F33CA70BBB83F2A2B63A2F29B0840FE83F03BCD53F73F216BDBDE37EC13408BC7DD02266F11C0801A8077157B0E402CFF802280EC16C01E06C3B1887323C0E5E9B665C1A1DDBF9D2614F01EE7F93F7E7B2AADC6050540DF86243A0F9614C029DC8BF69A03FF3F40C007040C83F2BF2EB72411D1A3B1BFF8AA85DDFCF30840B9B47DD981680AC084E432EB81AB03C0E2C1C34D70210840A72AE96C75A9F2BFFBC59A13E55902C029DA872C0146DCBF0CC0A5631BEE12402EB48D82A3B2FF3F598600EBC28201C07850D6E6D267E7BF8DB3A76CA13B10C03280791009B3E43FB00322E454F908C0CBE7A77F85F40A4000043F023A5202C0668B7A7DA81F10C0A0E68C475D650BC0803C86E165EC07C006547427CE2A0EC05614492F33C905C0591A384D936F11C0DDE916166592FA3F3B9BF448DFD3014004767407A599F53F9A08180DD86D0140A637F6779542FD3F6BC38449A3840D40C6AE011A13EED53FA5B3676D852D01C08BF49ED58F9D12C084FAB8FDCC4418400E8B65B228FB04C08BD9DCFD26EED63FDE5281BACFE5A4BF575ED1EAF11E0540331755AEF5EFE63FE3957C850C1DE93F429E0A1E1D81FBBF4E336D8ED3FD0940B6A16C8A80C002C0CAA141E430C70AC0B953F7AE23FCAABF8D5B7BFE6BB8F33F028BEB692A77FCBF465E77D928FAFF3FC4F24B7FDA2BE9BF2B22BEFD14D104C0025196574FA8DB3F3E85B8ADD55AACBF460618960629FD3F302298C5DDF0E5BF76BA04B7708F1640158C187EC1E8F03FACA5303F6CC80D40381525C0BB4ECC3F613004E9294B0040AF23A140303A05C03DCD01BCC80901C08CF310955725EABFC8D82CC19C62E7BF7903D610087A06C08A192345E86DECBFE47F195D96650FC0DC7BD74854C108408A3EB59F7B17FA3F67DE1D5B9A4DF3BF43F15E002A750E406AC3A1453E93144091ACE059DA73CEBF88CE83F0CD45F1BF63C3D7A2F559E13F637265E89580D1BFC806843F45DAF93F36E033146D37E63FFAFE5FA14637FFBF3F8E9E51A1CBC3BFA5FC9B4F9F7AE9BF7EF3BDC986A106C02A893FAA099EC7BF10EBA77FEC97E3BFE836B42DAF83DEBF1CAC9AFEC21206C0D142595FEC1A04C098F1682B229AF7BF3C6E1CED3BB8DF3F75FA613667CBEFBFECDAF1EEAEAEF63F5D9DE382AD90F03F100F934B98850840BAF0FC7C36340940E6CB477C493CF0BFFDF6F76D4276F63F88058AF9D10410C02E331711CB0D0E40AE9B21EBB845F4BFF3F16F72D2B8F33F56E5051B2B3DFA3FBA3547420A1719408C54142724D50BC0AA3F0D7B09F311406747BAD6BEEA00C0C0C8F2F92C6219C09A5E7313582915C0843147772D841040CB6807079E6806C00FC40B3AD4CA0540CD95FF8F958C10404C9C3DAFC39F0FC0060B1A892CC01540093DBF0457F90AC0133D6CF86909F7BF0A2DF882D85719C066B5F470C55006C0BEE29D9D608B15C0F8BAD96E3B61084086B72AD7159404C07A14F9589118F73F23F161A352FEA3BFF9F8F8EEC7C8EC3FE6A0E04C544911406B2C10F220480740EA76C132C46D0F4045AC8E905B611140FEE003C0B2970840F6A8B1E06CFCE33FB5A2E7CE041AF0BF65B90660B8571040FEA58382F56EF6BF6C873B46D031DCBF7242C3DBE5B802407E09A50A9FCBB13F77625E3C08D570BF9F8FAF1A73640A409992466CEDAD09C0AFC7AC56B13C0C40079DE0DFA6A506C0EBD86D41F1BA12C064C6844E4EA515407035877533D6FFBF63B2D5E19988F0BFCABC3A60760905C09A1628272B020140CAB380CBC42D0E402E9DD7F47572FE3FDBFE97ADBE8309C0905DC1CCEA38FC3F3E7B33249D780DC0F51CC501AD8EE0BFF895E4C48916FA3F1C36FE4CD9D5EBBF87EC974F22AA04C028BB2351ED4817C0708F4B9A2FE70FC0D4C3A814B030F7BF6CE140F0587714C0EA592AEAE4A00EC09250EFD930131B40BCA8C8B99916FE3F540069E38DBA0CC07056E3211BC9004029CBE9A2AFC2D0BF207E31BF958D0FC0808DCDEF04B008C0C6BB2BD21CB516C0C603BA1AEBFD0040026F25C3D98900C0A28C174C4C0909C0BAFA54F7F58EF1BF08999CD4925004C0E20A33F02E780B40450B6543E268FF3FC28FAFB7989EF03F75808C190AC70140C1A7C6E1534E05407639167290F1014020B38294A34E1AC08C498CF5C375D7BFCDE122D0DEDB05C01917E14B907C0A405ABE13532513EDBF9A6E56C232190D4026604E59A42B00C068708EED2F6390BF94EA6DF2EA3119C0A66C1F36A8120BC06C8CE7A9D554F0BFE40A2E8EB50107C0AAAE7400C253FE3FD9A803DF7ECA114014D841DCEDE41140E686C5D6EA8608C008DAFFBE467AF33F177DDC10A9F702403B90C0074F5BC33F43E849585CE2FDBFB553C8E601170040EA03A6C3AAF9A1BF0AC8C312C4D019C0016B0B0F5D9BEABFF347BAC2EF92E23F7ABA9A05CAE40B402208A2E5376901405CDD2FF39664EE3FA7527A370851EFBFB03C0C71F8A90A4036FD32A3531509C0A61AE8F9D8C8F83FFA2637E414C9FE3F85266510778EE1BF3A29789956F8EEBFDED37940BDD712C0FB666688E34DF13FEA75A34850B8024042A5E1C53A6B13C03C4DCC635BFAED3FAD7A69A02AC71B40E80FA0E6D20C1340AE52915A18F2FB3FC017AD71BBEAE3BF904320852841FA3F033912B9059D1540CF249FAC2E5FAABF2AADC97F4DF9EE3F7641D83AAEC0FFBF4264994DAC32D73F2A437E7DFAD7F3BF806A298B0A7FF4BF5A42F614B717E6BF18AEE34AA61009C03570476F900EC3BF69A2F15C6C40FD3FDE2E90FBB2291240B0E6248B8ACA0840F44978B5237316C0BF87064F7B15E33F5497159E742DF4BF006BA225DAB909C0665992BF95DCE1BFF2F598DD07F8A83F3C02C8D6E2F113C0D6EFA5AA7916D6BFDCAA329993FAFDBF714DEE38D73101C0BC9850BFCFF713C0474502B6213113C0166F49503CE802402A5258F1FA1B11406E77E372D39DE2BF5404E1416865FC3F7078A4771675FF3F83A62B0FBF8004C01E48F4A331DEF43F1CCB6D775B35F9BF1EB1AAE5F11CF2BF624FC202C4ECC8BFA36CD38ADA4B1040FB8C965150CBF33FAABB3F57B0BF10401C8F69ACC5F209400387F141503415C028E75979D07106402A03DD24A2DB114048A4D4557A7EE43F7C574A7E31B2BF3F849F00BEA28A00C06C3709E9E7420CC0E2FB459FBA92C93F0A3BE2188C9722C0DDBFD7123EB003405E3E80FF84140CC00C0520E2CDB1EBBF92BA7FCD265608C06264186762C501403661FD70B57D16C0D618AC0E568718C0ACEBE2885742F63F2C2DEE18E79EEBBF002D7ED90218EFBF345C619D76D6FE3FE928ABD754660940904386F9DB8206C0F286AE39476304407C53FE94E4BBB23FA62B7D43F8130240F6D6189A4B4214C0BAE42D993BC5084049ED7195EF0EBB3FA069C22608D6EF3F9CDBD39561B1E23F7A0129FE193A18C0042DEF1A055D02C05639BDE436140EC0CE188F47552A24C0DE36115DFBD8F73FE0B476214124FF3F19B7DD5CE930154085E06076088102C0967280FED6DA0CC0220D7CF0A22BAFBF322D950C0C7CBF3F2EA6CDE361CC03C03CE8FC5868B81140B71858183D1DEABF8F84CF6715A6E1BF7DFD1ED9D4150E403891F36188B8FD3F26577889C18BC4BF394D4BC3B35F07C0FAB33CBC70000A4012AB66A91FBB0F40F7C07A7176650040C887B4F386891040C64E0E19EDE40A402A2178168CD9E1BF3485318470F5E7BF4BE78E7CABCEFD3F4CCBDBC3E36F15C06C8E2A89E94016C02253C17FCBB0EE3FB5BA43633D7E913F0583315B819E05400A2863275F82E0BF280BFD609F5AC83FAB1EA0247F4BE43F2EC68C299BF6F1BF7E8411DF9588E8BFE8CCAB01F309F93F880F942A518CFB3F9EE97CE579A7FD3FF7061D8EE0D8D8BF49ADA7EC54331240DC51ECF63CF700C00ECE6337DAE6F13F"> : tensor<20x20xf64>
    return %cst : tensor<20x20xf64>
  }
  func.func private @expected() -> (tensor<20x20xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F94C63245420DDB3F000000000000F87F000000000000F87F000000000000F87FE8427DCC24A4DEBF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FF680B2924467EE3F000000000000F87F000000000000F87F37363370450FE1BF57B31D12C4CDF9BF000000000000F87F000000000000F87FBC4A508A4712EC3F000000000000F87F89A636944636E4BF000000000000F87F000000000000F87F000000000000F87FF73C9FBA7C31F33F000000000000F87F4A6C681B4C0DD2BF000000000000F87F000000000000F87F000000000000F87F000000000000F87FF8D5430D3EEDEABF000000000000F87F000000000000F87FC7A5EEFE4ED4014032B09A818D23B2BF000000000000F87F000000000000F87F0868BEECF097EABF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F7D1B3EF53FCFB83F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F00EA30782D09E0BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F392828F2FBAAB1BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FB0A25FFF145DDEBF000000000000F87F000000000000F87F000000000000F87F3408723C29D1EDBF000000000000F87F3EDC8D38CAA2E83F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F7082D870B4DAD63F000000000000F87F000000000000F87F000000000000F87F000000000000F87FAEE0F9CCA6FED73F47701904C9E8A4BF000000000000F87FDE7B5E5E1ED5EC3F4645B4207DECF03F000000000000F87F000000000000F87F000000000000F87F000000000000F87F0EE0D3EB8B02ABBF000000000000F87F000000000000F87F000000000000F87FDCF73317D5FFF0BF000000000000F87F808A67382B9ADD3F51DF24F24462ACBF000000000000F87F5B360268DCDEEABF000000000000F87F000000000000F87F000000000000F87F129B48F677C8CC3F000000000000F87F000000000000F87F000000000000F87FC14AA611D05DF2BF9DD25416F7C5EDBF000000000000F87FD3DD4EEF37A1F6BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FE45DC913250CCFBF000000000000F87FE1FCFC37176FE33F8BB0385A94F5D1BF000000000000F87F26E89E218965EB3F000000000000F87F4892D1039FF4C3BF4C3D06943F69F1BF000000000000F87F56FD04BA15E4C7BF3D58177D7FCDE6BF6E7D7C99289AE0BF000000000000F87F000000000000F87F000000000000F87F932741F43564E13F7330FAE6E7F306C0000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FB2918C42ED00A4BFAD1566C0A883F73F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F698068CA4070E73F000000000000F87F000000000000F87F000000000000F87FEB67F9090344DEBF000000000000F87FFA5825F7FAD2B13F90AB05720ED570BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F217BC5A76B54E2BF000000000000F87F6B486971D851F5BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F30C89F1F0529D1BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FD61B78DCC59AD8BF000000000000F87F000000000000F87F7031C4DF524FF8BF000000000000F87F000000000000F87F189D3BA08B6390BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FDC5FE8059C81C33F000000000000F87F000000000000F87FEFB9181F8FFBA1BF000000000000F87F35C87480F915F3BF549033444938E53F000000000000F87F000000000000F87F061109D8B545FD3F3E6B7521691D02C0000000000000F87F000000000000F87F000000000000F87F000000000000F87F17166330D5B9E3BF12D385DBE27300C0000000000000F87F000000000000F87F000000000000F87F000000000000F87F8F411405A161FB3F000000000000F87F000000000000F87F000000000000F87F208F4AE04E53E7BF000000000000F87F000000000000F87F7F903D842965AABFA7D9EC32B3770040000000000000F87F9E5D404D664DD83F000000000000F87F000000000000F87FEF7243C5A128EBBF000000000000F87F4962E9DD1833C3BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F9257DDD4FBFFE53F000000000000F87F000000000000F87F3C796F76712AE4BF49B8F6F11AFDA83F000000000000F87FE2BBBF2A8608D7BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F50553D3CBA48E5BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F34548AE54C3FC9BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FB568A0610E49E83F5E1C28970ADCBF3F000000000000F87F000000000000F87FD05379BCFBEBC93F000000000000F87F000000000000F87F000000000000F87F8B1A851DED08F5BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FC889CC7891E3F4BF2E05A7F2F1F800C0000000000000F87F000000000000F87F000000000000F87F000000000000F87F315002487BC4B23F000000000000F87F000000000000F87F000000000000F87FA8982FD4E728BB3F71537A0ABCDB07409010017D5966E53F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F8123D0538435AFBF348D88CE0EA5BF3F000000000000F87F000000000000F87F3F9C834EA751F2BF8CF796EDB2DBE3BF000000000000F87F000000000000F87F64C1EFDDA3B9C4BF000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FF8F098330826E4BF008ABDE55C0AEFBF000000000000F87F000000000000F87F000000000000F87F40F37CF1FCF2FE3F1A79EBEDAC7E913F000000000000F87F1C5228C7A243E2BF0FC0BD8888A7C83F330E7C5533F3E73F000000000000F87F4E2AB43EFF31F0BF000000000000F87F000000000000F87F000000000000F87FC540F335E538DABF000000000000F87F000000000000F87F000000000000F87F"> : tensor<20x20xf64>
    return %cst : tensor<20x20xf64>
  }
}