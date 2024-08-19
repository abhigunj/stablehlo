module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<"0xE483A23E2F1E4A3F235378C056550340493AD8BF4017A0BFB974AFBE0528273FF70C9EBF742F9840F4A6733D1BFEB440393120C0E35FCC3EBA7CE2BF9F4202408D3405BFF01F093FAECB3CC05AB8EABEE5C446C09D87DCBFD4CDDD3FEC3D27BF397454C0CCF78D402F8271C0A5977BBF03A8C140A41C0E3F160D343E3054E7BEBF8029C0A87BCF3FCA3A4FC0050C3C400DD20D403AE732C021631DC06D4C8FC0FD245940EB5C8EBDBD1B4EBF88C28CC0568E1AC0AAA889409741FDBF1941A33F9BFE95C018AECAC0258088BF90EA8AC01BFC55BF8DFF96401C38953F55E145C0DFA7793F07265FC0608BCC3FF4E852C041B082C066BB163EAE704C3ED9D2F6BE735861C0F4706AC07E4E0BC067760840087A86C09DFE0EC0BB3F7E3FA6BCC5BEF02741C023B99EC0B70E0540636E0541B8C342C05FE34C403B1507C156BA663F33D8F0BDC361443EBDEF11C036E01040767EE04007F546404678CDBF063455BC91A887BE52DFBCBF7B720FBFAE83AC3E37A96340FFE0A4BF86B90A4091A137BF1EC3193FAE600EC0A3E044C05DC121BEF62CE5BFBA7654C09210313FDF23D3C07196FABF45D718C08B677BC074D545C04CE6E1404A135AC0435C25BFEF138DBEAAACF840D1428840426EF3BED053B0BF5FC3CD40EC2600C05D7C62C0C0890DBF1130E63FB428853F7FAFA3404985CCBF9BE229BE70B395BD5D7BD5C03F29953F56B72F404113A9BF9B2802BFC1C1CFBFBFE8ADBF1C5BF43F3438DBC090B2993F8CD81EC0B519AC40B9F249BE51DBEABF387483C01E9104409DA7083FF9B4DA406FCEA1BE6BC9AB3FFC27E6C0395E35C0205788C0FD595E402C50593F34A20BC092E83C402788BA3F937E4FBE1DD9E0BDBB881E3F622DA3C05D3BDC40CFD698BF7D8DA1401CE95740E0FD1940164E84C0B372A63F355B1E3FAEB34C401877C23FFD0F0340ADA9D63F2350AA40BA05A3BF07D02B4078C2A940BF9886BE9744B2C0624BDBBFC31382BF8933CABEB821EFBF64C67B3E9D6D26C0D82ED0BFEC8E0140632CA83EE22C534085131A3FE0DDE2BF04568ABFCD0186C031753740B1BC7B3FB055533F824904C0CDE9B1C0F37A1D40A896C0BFE68EC9BF6494A8BF3346E33FF67019402CEF65BF58D371408D9642C0310E1740FCC11EC02BC8EA3F9C4A794000154FBF0FC55540EE4984BE7B27A2C0ED387CBFEF41463EBD9DE33E629A23C034235CC00DA8F03FB48FCD3F724CABBFF17F21C0E167973B68E93CC010349AC0768E0EBFDF1EE93F833DB740A4B39DC03CF623C01C1D114024A699C020FF78BF040DFC40ED869FBF40CA963E0937EA3EFCBB8DBEA410F5BF374F2EC0A67482C030FE863FBF67FCC0F7E180BFF5DF57C0F6732B4017D54F40FAE94B40B72DB43FD1E59AC0674064C01FE465C01C9121BC00B7863F141D653F649416C03A0A7840A06496C05F0169BFA4C476BFBB73563E7B3C2F40D8205AC0E618C5BF6C43E5BF40D55BBFB5CBA7C0220DC13E0ED683BF9602CF40B2FBC740EE4D373F82DDA33E9670EAC0353283C09AC537C04CFED1BF56184D403EFC87BF1BB6A2C0D62CD9BB90B55040B7B4B5BF143135C0A0BF11C0125B73C0AC1F29C0F48BF53D69299D40DC4244402E8533C0FA8FF6BFCCA90841972477404F2FC6BF255E2D405AD68B40C04049C0AB682140BBCD9E3D6360AD3EBB9AC2C09E1D523F714F6BC08F5D8F3F13A25E409941FDBF0D3E8140D3056D3E8D3725C0BF3D96C0533323BFA3DC01413A597F40A689493F2A318D3F91203640104B8F3FFF4E2CBFC53FF23F516D8040F21B9E4089C84BC0BE34D0C0EA874B406C68384001722940FFE73EBFD889E3C0B9445E40610BB33E5CD78ABE1B897440641913BF80CBDB3F8AD3974070D667C02222C03F3E1002C09D9CA8C022C6B2406E93D63F417739C0E17C44405256B4BEC0B588C0038CBB3E5B7B2BC0CB618ABF4B42C9BF013024C08C13B9BDCCE8DE3FE9A8FFBF2F0D55406B00CA40BFE3123F21A281BF5DB570BFB52CC3403234FF3FD43BC53E7919763FA103D5BF0E9B71C06562A6BF745F34401DF3693F3296403F337F34C0AEC333BF28C202C002F567C0CEC61E40FA4B934017F25A3E924950BE1940B9BF065375C0CF5F543F08E090C056038AC09CDA5FBFE743814023BF963E8CFC00C0C8623440BBF4123FD4E7F1BED3BCDFBF4B43704091F687C075AF08BFA466A9C0671B76BE65B493401A3EDE3F2C9737C06D7FDC3FD6EAED3D8EFC52BF3D5BA73F26B2E13FEBD7774065229140678D1EC0B1C0F63E2D95A84055B5F93F6E9EA0409C660140F38BC33FE48AA040FBB6383F3ABB313F1A960AC0E55DBCBEC9AE41C03F2E7DBF5558BBBF5C292ABF9C878E40A45C7240B6F9F33F125AA1BF42A19E3FDFDA3EBF10E91F4067BD4F40ACEADE3FB62F10C0891B23C030037CC0157D02C07FC2F2BF324C2D3E3AF80B3F3072733FC30510C04C26F23DDB7262C0D9C68CBF2B330240B31A1D40D25E503DABA2F83F6768F73F41D9D13FA3EA85C027CEFEBF4CC1D1BFA7D08BC0822F54C094CD324016015A40B36F8F3EC60887C0256F6B3EE706B63FD99E1CC0C2FBEEBF76DD7E3E1007F53CDAC7CDC002B98AC02727ED3F7C202540C4204840ADCB52C0E10A913EC90469BF63402740E3E5C83C01822B40B19D4140FEFE1A40AE99C23FE9E440C0D0B981C0841CBF3FD8F942C0411599C0B5DB0B3FB3BD82BFDA827640AF2A974027126BBFCE1290C069611EC0EB50733F76B3DD3FC6216FBFD915FEBB1F6740BF876A3AC0C868AABEFE0CF3BF255FFDBF87134BC0E37363406D2A7BBFABC4AA3F9F7B15BF0AE8DABF2588F3BF357579C0B26B1C40E7E87B4045311540E32994BFDA2A86BF427B453F920EE93F35C622C04CEE78C0238ACA3FD194C8BF7EB496C07904003F6DA05B409A84A93F68B31E403AAF2ABF328B0AC0EEF15D4024BC3BC089E2CC3FF9A9854070F70640FDB249C0D6F7E53E07682F40EC3F2A400491C0BFE3D2FABFB6F4A5C0434CA0C0398695BF3AA0644066EF06BEEAA00FC1"> : tensor<2x3x9x10xf32>
    %cst_0 = stablehlo.constant dense<"0xEBE9243FE1AF46BDFD630C41ECEE5A3FA3420A4002D33240EE0EA540BB8B16C017F8D7C095FBC6C011A3EF3DC95A06BFBE67C83F8D8E65406CB232BEA33CBFBF41E43AC0CAFC904083BE973F5FD203C0A88D5DBF70FAB4BF9C96063E4BD67DC047AB4E40220BCA3F77EB11401349A8BFCA24383F7DCBCDBFB521C1BF6A0F64409C4BC5BFAF10B93F73598FC0BD8F60C05AEC85C0F71A3B402095A0BF1A10FDBEE5B65C3FDF5313C072648EC0B60081BF2818EEBFFA7470C0E76E03400E471B40856B01C0AB003C40B2D7C6BF494EA53F603930C0E99D664072C9BF406AC480C005A5A2BF58B007BF50B4933FE9000340433538400606CABE7EA3953FB6CFAB3F6E642E40F4131040051091C0616493C0AC104340BFD8C63F3BC94FBE9B41EBBF615306C0D6370640D7A25ABF8B3FDA3F2F68BFBF60BA83BFE6D481C0A5CDC2C0D8CF1FBFA6E10B401A29FF3B8F9D4B402E238C3F3BD0D540C90E113FB4354C40587328C044FB17BEEE6E6F40D5559A40509C8DBF6D853A40049F4D3FA6B69EC000DF2D3F2233AEBE97B6D23F50B327404A3B123E3D25D7BF5C9B914023F03D408559563F11C2E63F67DBEE3FD1E21B4086BCBABF4C6911BFDDA1B9BF6B366CC0491BBDBD3B04CB4092E125C0DE7974C0AF6F8DBF6C2A2440FE862940DC368A3FCFB1C33E11CB0440400A0F4064DA62C037169E401E3B79C0F9484740D47D19C0DA36763F67D3F13FCCBCD3BE50ACFD3F5C310ABFB4368C3FC75648C0421FA7BF11E0FB3F638FB640429E8E3F86713EBF8CE95FC015DC1FBF199153C0230E403F03A73140F4231E3FD1BBCABF4CF8C040565F083EB9D65B405661DB4045DD24C0C95957C08C3E81C0AABD24C01E3EAA3F943A1EBE2F3CE3BF1FE5063FBCA6D0BFEADE744023D5A8C06CC4BDC0B65F313FC5E06DC04402A7C0A1B337C018EA11C0210926C0407BECBF9FB060402DFBAEC0D712BA3F83C58040E61B4ABE768DE5BF2CD37F4031674A4065D2CA3EC78401C0"> : tensor<3x3x4x5xf32>
    %cst_1 = stablehlo.constant dense<"0x12FB2E41D8EB1741CEE6BD40CF793841C574DE4050C335414751D640547C254192442C414D9D4041DCA407418DF841414072F1409EDC00418B654741C1BBEE4057A21A4116B41E415DEE044115212441560F20419A2311410CAF4441C607D9403FDFF640D5C522418F1E37418DF841415DEE044194D726415B5B0A4108F65441C5745E4194D726411747194112FB2E4194D7264187AC574119DA134109894F413D4CFC4094D7264157A21A414D9D40414751564197FD1B41515630410A1C4A414E303B415B5B0A41FE2E804192442C415DEE044194D726419BB60B418DF84141CF79384183F367418B654741C4E16341CBC0484108F65441033D6541FE2E8041C607594112FB2E418AD24C4110683441883F5241138E29413FDF7641CEE63D415AC80F41BF28F4404E303B413D4CFC4097FD1B413FDFF6404E30BB404105EC40D6581D41442BE1400D423F41D5C522413D4CFC409C490641CF79384191B13141152124419EDC0041DE3782401D930341DB110D4108F6544153E92A418F1E3741DE370241D10C3341CF79384146BE5B4110683441956A2141D97E12415DEE0441C4E16341D97E124117471941560FA040D8EB17418F1E37418486624194D72641D5C52241BE95F94092442C4149E45041D5C5224194D72641BC027F41138E2941D29F2D414A774B41BF28744149E450418E8B3C418DF841414C0A4641C24E694109894F41CEE63D410A1C4A4157A21A410CAF44414A774B418F1E3741FEF07A4108F6544191B13141CBC0484116B41E418F1E3741C92D4E4116B41E41D3322841D33228418AD24C41C89A53414E303B410017704194D726410FD53941C1BB6E41033D6541C4E16341BCAD894146BE5B4105D05F419EDC80411A6D8E415B5B8A41883F524143986641BE957941383E974105D05F41C6075941442B61414C0A46417C5288411C008941C1BB6E41FF8375419EDC80415AC88F41442B614147515641C89A5341C4E163418B65474194D72641989016411521244184866241138E2941D10C3341BC02FF408E8B3C418B654741D97E12418B654741547C254150C3354185195D41CC53434143986641FEF07A41BC02FF40C607594100177041BF287441BC027F4100177041883F52418AD24C410CAF444109894F410D423F4197FD1B41CBC048411068344187AC574119DA13418AD24C4110683441547C254194D72641"> : tensor<2x3x6x6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x3x4x5xf32>) -> tensor<3x3x4x5x!quant.uniform<i8:f32, 0.0039212212843053483:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<2x3x9x10xf32>) -> tensor<2x3x9x10x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %2 = stablehlo.transpose %1, dims = [0, 2, 3, 1] : (tensor<2x3x9x10x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<2x9x10x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %3 = stablehlo.transpose %0, dims = [2, 3, 1, 0] : (tensor<3x3x4x5x!quant.uniform<i8:f32, 0.0039212212843053483:-128>>) -> tensor<4x5x3x3x!quant.uniform<i8:f32, 0.0039212212843053483:-128>>
    %4 = stablehlo.convolution(%2, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x9x10x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<4x5x3x3x!quant.uniform<i8:f32, 0.0039212212843053483:-128>>) -> tensor<2x6x6x3x!quant.uniform<i32:f32, 1.537725862903607E-5>>
    %5 = stablehlo.uniform_quantize %4 : (tensor<2x6x6x3x!quant.uniform<i32:f32, 1.537725862903607E-5>>) -> tensor<2x6x6x3x!quant.uniform<i8:f32, 0.084777487960516234:-128>>
    %6 = stablehlo.transpose %5, dims = [0, 3, 1, 2] : (tensor<2x6x6x3x!quant.uniform<i8:f32, 0.084777487960516234:-128>>) -> tensor<2x3x6x6x!quant.uniform<i8:f32, 0.084777487960516234:-128>>
    %7 = stablehlo.uniform_dequantize %6 : (tensor<2x3x6x6x!quant.uniform<i8:f32, 0.084777487960516234:-128>>) -> tensor<2x3x6x6xf32>
    %8 = stablehlo.custom_call @check.eq(%cst_1, %7) : (tensor<2x3x6x6xf32>, tensor<2x3x6x6xf32>) -> tensor<i1>
    return %8 : tensor<i1>
  }
}
