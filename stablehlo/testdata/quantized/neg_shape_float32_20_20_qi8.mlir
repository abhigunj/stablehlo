// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<"0x942A303F3866C83F5451B03F2C8038C056E615C0BBFB9E3F0BC8CF3FADD71EC0C154B9C0868987BFCB136B409DB9FBBEAED4C3BF85245A405B06B6C084B18BC05476A7BEC56082BE2B2DA1BFE1828240B17F4DC087323740561827403AD2DA408F6026403EFF55BFD66F93BFC7BC52C0A6420FC062CDAEBFE2FF1CC1510191BD26CEF83FCF4A313F6B46673FF79A2640825F5240E19C1A40BCD533C01E0A823EFE587C40A0E3BEBEF5DAD73EAC992F3E1646DB3E1881D53E387DE23F0E27A43FB0749840A2177D3F17C7E6BE8D2C14C0DBA0D53FAB762D403E2E64BF9B0B05C0F81AB5BF313B294038F7DF3F13FEDBBE87DC93C0042BCABF3026E5BF3E9A37C0A972DDBE39FB99BFC3F0DABEC4100A3EB7549C3D5815AC3FD40339C0F2CBE73E4DDACABF43D4AF3F15812B3D7DA1813E369BB740D2BC65C0A284EC3F352245C0FAEB5CBE7250AE3E3CBCA5408004773FBD449240A341A1403A5B36408D830FBF87CFA9BEF2141340782D19C061CB0FBF41818C40A3215BBF39E53FBFA1004740BBF2B8BED1368DC04037F03F282FA73F14E395408FB34F3A773207C0ABF9DB40B4DE8ABE0791283FA1772B409989DEBFEAABDABECCFC68404BABAF40F27B8C3F373976BF7E1BC540F5910A3F1ED2B4C05F5045C06232C3BF3F4AB33FDE440DBFA2BE0A3C26FA55BF8A37153FB1B9BC3FA250813E2FCC003F9550ABBEBD60B8BF7B60A7C04C915ABE99C45040E3A6EA4019AAAA3E5AD48040F2FC2FBF6D002A40F06DBD3F1377543F768212C0788457BF755435C0D68DE6BF679788C0E3278640EEC134403DC96C3F5285853FFA62E23F03C41DC0E841E1BFBA67BEBEF1018D3F6A5ABE3E213EC5408C726640707A98C0A06916C017B45240D138E8BF03DC87C04AF55E406A6A373E891799BFC8BDA940BDBF17C1F70E434093DB5BBF823C1440281DD2BF0ED43DBF5F774140B899323F1FA46540C6EBE13F9D52B5C0A9AA973E8A0E133FEFCC0BC09761F93FDC936B40CF46AA3FC3D0AFBF6036B4C0E2628240BB713B40ECFC25BF4163664062AEA9BF3EEF684077172E409D68F63F8F54CA3F52900AC0218AF6BF7848A6C0761DA9BDFEA0B44072CB5FC06F619240D47BA3C09ABEFBC0D8A36D3FC8F08EBFC18813BF9800174043F92D4011992DBF294DDEBF2CA71EC06BD674BF4E4C6F40067692C052AF42407E5F4840EF0F34BE26A210BF82858B40400EF53D64D3A4BF8099DF3E529E3540179279C0204753C0B7B1613F76C50DC01A1243C0DCCBC2C04065A2BDD674253F59B6A23E315BF3BF052A6BC09D30FFBD9E068F4098F259BCCE24BC3F38F38A3E6B64D34035F78FBE57FAB73DF096A1BE394E93BFEA358BBEDFA6D43E3CED6440EE9773403418E53FFF87F240FA83DE3EE3972B3FEE3E413FB6C94E406D898FC0A8C891BFE5414ABFCED98CBF00DCD4BED3FF5540FAEE853F1B165BC0ED258D40ADE3F1BF6E439A3F29C2BB3FAA88C73FB5FD7AC0E080EBBE04A66A3F35BCE83F2610BDC094762840C4CA784090F1ECBFE00CB040FF8E78BF467B9C3FD5D5AEC01B5BE03F99FC3FC09A3A6CC05C109E3F72FB88C0346E753F81CC1BC0EEE4E0BFECBCE0BFE3134EBFBE6DFCBFE408C7C0DDD78240C56231400CFB1540CA75BE3F059542BF97A47840F52781BF7BCD41C03D3586BFFD332A3FA6395CC092180F405303DDC01FB9F03F62FB3BC0512A94BE70C6B23FD328C2BDE415CDBFFEBD38C094FFD8BC9E7395C0AF39ADBF9773CFC070945B40AC71BEBFB3EA2A40B1716AC08ED742C0BB359BC09C36F83F4A04A73EA6CE2BC0524E96C0EE244BC0BC0CAF406705324038FC943F025E93402BB2E1BEC1D7B13ED4183CC0A028983F737CAB3F93D66A40F984BFBFDC8F88BF2008A640A4D479C0136154C093377940A2E60540E9BDA33FF43DD8BF69419640525B50BF6AA8EC3F5FFBEABEDDBBD4BEB18B24405A4710BFBEC70DC0DAEFB83EADC1AFC03CE219C0AD2A89C0734CCC3F3D34D2404639BE3F1CFB1140833787C02A5113C025AA36BF330271BFF34D274093BCA53E69C251BC95B79BC0C7FA3BC0CC4A7540243743C059ACD8BD0A5D72BF8722963F7AD7683F4DA1753FB750733E8ADDDFBFAB50E1BFF77491C0320DD83F97B82AC0C46E2A4055C1013F2C79433F0F749F40325162BEB51F294034D8FEBE09EC0CC0FC55B6BF743814C0ABA9913E322F823FA1B394C0FAFD9BBEBAF0123E6742D1BF4DE0033FC9381EBE4266B1C0"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<"0x74AF2FBFA9FF7FBFA9FF7FBF0000000000000000A9FF7FBFA9FF7FBF000000000000000000000000A9FF7FBF0000000000000000A9FF7FBF0000000000000000000000000000000000000000A9FF7FBF00000000A9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF00000000000000000000000000000000000000000000000000000000A9FF7FBF76B131BF99E666BFA9FF7FBFA9FF7FBFA9FF7FBF00000000568282BEA9FF7FBF000000008FD8D8BE75B030BE91DADABE8DD4D4BEA9FF7FBFA9FF7FBFA9FF7FBFA7FC7CBF0000000000000000A9FF7FBFA9FF7FBF000000000000000000000000A9FF7FBFA9FF7FBF00000000000000000000000000000000000000000000000000000000000000005A8808BE659898BDA9FF7FBF0000000099E6E6BE00000000A9FF7FBF75B030BD568282BEA9FF7FBF00000000A9FF7FBF000000000000000074AEAEBEA9FF7FBFA3F676BFA9FF7FBFA9FF7FBFA9FF7FBF0000000000000000A9FF7FBF0000000000000000A9FF7FBF0000000000000000A9FF7FBF0000000000000000A9FF7FBFA9FF7FBFA9FF7FBF0000000000000000A9FF7FBF0000000070A828BFA9FF7FBF0000000000000000A9FF7FBFA9FF7FBFA9FF7FBF00000000A9FF7FBF5C8A0ABF000000000000000000000000A9FF7FBF00000000558000BC00000000639515BFA9FF7FBF558080BE558000BF00000000000000000000000000000000A9FF7FBFA9FF7FBF71AAAABEA9FF7FBF00000000A9FF7FBFA9FF7FBF8DD454BF0000000000000000000000000000000000000000A9FF7FBFA9FF7FBF9DEC6CBFA9FF7FBFA9FF7FBF000000000000000000000000A9FF7FBF7EBEBEBEA9FF7FBFA9FF7FBF0000000000000000A9FF7FBF0000000000000000A9FF7FBF7AB838BE00000000A9FF7FBF00000000A9FF7FBF00000000A9FF7FBF0000000000000000A9FF7FBF76B232BFA9FF7FBFA9FF7FBF00000000659898BE619212BF00000000A9FF7FBFA9FF7FBFA9FF7FBF0000000000000000A9FF7FBFA9FF7FBF00000000A9FF7FBF00000000A9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF00000000000000000000000000000000A9FF7FBF00000000A9FF7FBF00000000000000009DED6DBF0000000000000000A9FF7FBFA9FF7FBF00000000000000000000000000000000A9FF7FBF00000000A9FF7FBFA9FF7FBF0000000000000000A9FF7FBFA5F8F8BD0000000093DEDEBEA9FF7FBF000000000000000095E161BF000000000000000000000000000000006EA525BF6CA2A2BE000000000000000000000000A9FF7FBF00000000A9FF7FBF5C8A8ABEA9FF7FBF000000007AB8B8BD0000000000000000000000008DD4D4BEA9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF93DEDEBE72AB2BBF80C040BFA9FF7FBF0000000000000000000000000000000000000000A9FF7FBFA9FF7FBF00000000A9FF7FBF00000000A9FF7FBFA9FF7FBFA9FF7FBF00000000000000009BEA6ABFA9FF7FBF00000000A9FF7FBFA9FF7FBF00000000A9FF7FBF00000000A9FF7FBF00000000A9FF7FBF0000000000000000A9FF7FBF00000000A2F474BF000000000000000000000000000000000000000000000000A9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF00000000A9FF7FBF00000000000000000000000071AA2ABF00000000A9FF7FBF00000000A9FF7FBF0000000000000000A9FF7FBF00000000000000000000000000000000000000000000000000000000A9FF7FBF00000000A9FF7FBF000000000000000000000000A9FF7FBF6EA6A6BE000000000000000000000000A9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF0000000076B2B2BE00000000A9FF7FBFA9FF7FBFA9FF7FBF0000000000000000A9FF7FBF0000000000000000A9FF7FBFA9FF7FBFA9FF7FBF00000000A9FF7FBF00000000A9FF7FBF0000000000000000A9FF7FBF00000000000000007AB8B8BE000000000000000000000000A9FF7FBFA9FF7FBFA9FF7FBFA9FF7FBF00000000000000000000000000000000A9FF7FBF6EA6A6BE000000000000000000000000A9FF7FBF000000000000000000000000A9FF7FBF9AE868BFA3F575BFA2F474BE000000000000000000000000A9FF7FBF00000000A9FF7FBF568101BF81C343BFA9FF7FBF00000000A9FF7FBF00000000000000000000000000000000619292BEA9FF7FBF0000000000000000629414BE00000000578303BF0000000000000000"> : tensor<20x20xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<20x20xf32>) -> tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.negate %0 : (tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:127>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:127>>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
}
